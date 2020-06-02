volatile int x;

void __attribute__((noinline)) sink() {
  x++; //% self.filecheck("bt", "main.cpp", "-implicit-check-not=artificial")
  // CHECK: frame #0: 0x{{[0-9a-f]+}} a.out`sink() at main.cpp:[[@LINE-1]]:4
  // CHECK-NEXT: frame #1: 0x{{[0-9a-f]+}} a.out`func3() at main.cpp:16:3
  // CHECK-SAME: [artificial]
  // CHECK-NEXT: frame #2: 0x{{[0-9a-f]+}} a.out`func2()
  // CHECK-NEXT: frame #3: 0x{{[0-9a-f]+}} a.out`func1() at main.cpp:25:3
  // CHECK-SAME: [artificial]
  // CHECK-NEXT: frame #4: 0x{{[0-9a-f]+}} a.out`main
}

void __attribute__((noinline)) func3() {
  x++;
  sink(); /* tail */
}

void __attribute__((disable_tail_calls, noinline)) func2() {
  func3(); /* regular */
}

void __attribute__((noinline)) func1() {
  x++;
  func2(); /* tail */
}

int __attribute__((disable_tail_calls)) main() {
  // DEBUG: self.runCmd("log enable lldb step -f /tmp/lldbstep.log")
  func1(); /* regular */
  return 0;
}
