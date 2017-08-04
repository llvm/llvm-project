// Compile with $CC -nostdlib -shared TestModule.c -o TestModule.so
// The actual contents of the test module is not important here. I am using this
// because it
// produces an extremely tiny (but still perfectly valid) module.

void boom(void) {
  char *BOOM;
  *BOOM = 47;
}
