void noop() {}

void fun() {
  int user_command = 474747;
  int alias_command = 474747;
  int alias_command_with_arg = 474747;
  int platform = 474747; // built-in command
  noop();                // breakpoint 1
}

int main() {
  fun();
  noop(); // breakpoint 2
  return 0;
}
