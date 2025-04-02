void func() { __builtin_printf("Break here"); }

int main(int argc) {
  func();
  return 0;
}
