// Test that the lldb command `statistics` works.

int main(void) {
  int patatino = 27;
  //%self.expect("statistics disable", substrs=['need to enable statistics before disabling'], error=True)
  //%self.expect("statistics enable")
  //%self.expect("statistics enable", substrs=['already enabled'], error=True)
  //%self.expect("expr patatino", substrs=['27'])
  //%self.expect("statistics disable")
  //%self.expect("statistics dump", substrs=['expr evaluation successes : 1', 'expr evaluation failures : 0'])
  //%self.expect("frame var", substrs=['27'])
  //%self.expect("statistics enable")
  //%self.expect("frame var", substrs=['27'])
  //%self.expect("statistics disable")
  //%self.expect("statistics dump", substrs=['frame var successes : 1', 'frame var failures : 0'])

  return 0;
}
