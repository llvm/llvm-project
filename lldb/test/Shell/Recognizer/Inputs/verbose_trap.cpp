struct Dummy {
  void func() { __builtin_verbose_trap("Function is not implemented"); }
};

int main() {
  Dummy{}.func();
  return 0;
}
