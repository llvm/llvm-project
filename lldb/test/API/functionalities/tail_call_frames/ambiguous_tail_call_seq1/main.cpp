volatile int x;

void __attribute__((noinline)) sink() {
  x++; //% self.filecheck("bt", "main.cpp")
  // CHECK-NOT: func{{[23]}}_amb
}

void __attribute__((noinline)) func3_amb() { sink(); /* tail */ }

void __attribute__((noinline)) func2_amb() { sink(); /* tail */ }

void __attribute__((noinline)) func1() {
  if (x > 0)
    func2_amb(); /* tail */
  else
    func3_amb(); /* tail */
}

int __attribute__((disable_tail_calls)) main(int argc, char **) {
  // The sequences `main -> func1 -> f{2,3}_amb -> sink` are both plausible. Test
  // that lldb doesn't attempt to guess which one occurred.
  func1();
  return 0;
}
