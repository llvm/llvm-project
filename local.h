inline void bar() {
  auto foo = [](auto) -> int {
      return 50;
  };
  foo(50);
}
