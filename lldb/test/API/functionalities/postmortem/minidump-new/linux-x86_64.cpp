void crash() {
  volatile int *a = (int *)(nullptr);
  *a = 1;
}

extern "C" void _start();
void InstallBreakpad();

void _start() {
  InstallBreakpad();
  crash();
}
