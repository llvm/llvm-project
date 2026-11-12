extern "C" void call_func(void (*ptr)(int a), int a);

extern "C" void func(int arg) { }

int main(int argc, char **argv) {
  call_func(func, 42);
  return 0;
}
