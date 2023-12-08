void InstallBreakpad();
void WriteMinidump();

int global = 42;

int bar(int x) {
  WriteMinidump();
  int y = 4 * x + global;
  return y;
}

int foo(int x) {
  int y = 2 * bar(3 * x);
  return y;
}

extern "C" void _start();

void _start() {
  InstallBreakpad();
  foo(1);
}
