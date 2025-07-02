from dotenv import load_dotenv

from ava import  db, Ava

load_dotenv()
import os
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen

# Authentication logic
VALID_USERNAME = os.getenv("USERNAME")
VALID_PASSWORD = os.getenv("PASSWORD")


def authenticate(username, password):
    return username == VALID_USERNAME and password == VALID_PASSWORD


class LoginScreen(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Correct

        self.orientation = 'vertical'
        self.spacing = 15
        self.padding = [50, 50]

        self.add_widget(Label(text="AVA LOGIN", font_size=24))
        self.add_widget(Label(text="Username"))
        self.username_input = TextInput(
            hint_text="Enter your username",
            size_hint=(1, None),
            height=40
        )
        self.add_widget(self.username_input)

        self.add_widget(Label(text="Password"))
        self.password_input = TextInput(
            hint_text="Enter your password",
            password=True,
            size_hint=(1, None),
            height=40
        )
        self.add_widget(self.password_input)

        self.login_button = Button(
            text="Login",
            size_hint=(1, None),
            height=50
        )
        self.login_button.bind(on_press=self.authenticate_user)
        self.add_widget(self.login_button)

    def authenticate_user(self, instance):
        username = self.username_input.text
        password = self.password_input.text

        if authenticate(username, password):
            self.parent.parent.current = 'main_screen'  # Access the ScreenManager
        else:
            popup = Popup(
                title="Login Failed",
                content=Label(text="Invalid username or password."),
                size_hint=(None, None),
                size=(400, 200)
            )
            popup.open()


class MainAppScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.add_widget(self.layout)


        self.ava = None
        self._is_initialized = False

    def on_enter(self):
        """This gets called when the screen is displayed"""
        if not self._is_initialized:
            from ava import Ava  # Import here to prevent early initialization
            self.ava = Ava()
            self.layout.add_widget(self.ava)

            # Start  functionality
            self.ava.start_listening()
            Clock.schedule_interval(self.ava.update_circle, 1 / 60)
            self._is_initialized = True


class MykivyApp(App):
    def build(self):
        self.sm = ScreenManager()

        # Login screen
        login_screen = Screen(name='login_screen')
        login_screen.add_widget(LoginScreen())
        self.sm.add_widget(login_screen)

        # Main app screen (initially empty)
        main_screen = MainAppScreen(name='main_screen')
        self.sm.add_widget(main_screen)

        return self.sm


if __name__ == '__main__':
    MykivyApp().run()


