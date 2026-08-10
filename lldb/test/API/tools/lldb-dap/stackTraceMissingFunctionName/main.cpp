int main() {
  typedef void (*FooCallback)();
  FooCallback foo_callback = (FooCallback)0;
  foo_callback(); // Crash at zero!
  return 0;
}
