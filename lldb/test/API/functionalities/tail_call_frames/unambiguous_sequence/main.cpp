volatile int x;

void __attribute__((noinline)) sink() {
  x++; //% self.filecheck("bt", "main.cpp", "-implicit-check-not=artificial")
  // CHECK: frame #0: 0x{{[0-9a-f]+}} a.out`sink() at main.cpp:[[@LINE-1]]:4
  // CHECK-NEXT: frame #1: 0x{{[0-9a-f]+}} a.out`func3() at main.cpp:26:3
  // CHECK-SAME: [artificial]
  // CHECK-NEXT: frame #2: 0x{{[0-9a-f]+}} a.out`func2()
  // CHECK-NEXT: frame #3: 0x{{[0-9a-f]+}} a.out`func1() at main.cpp:35:3
  // CHECK-SAME: [artificial]
  // CHECK-NEXT: frame #4: 0x{{[0-9a-f]+}} a.out`main
  // In the GNU style, the artificial frames will point after the tail call
  // instruction. In v5 they should point to the instruction itself.
  //% frame1 = self.thread().GetFrameAtIndex(1)
  //% func3 = frame1.GetFunction()
  //% func3_insns = func3.GetInstructions(self.target())
  //% self.trace("func3:\n%s"%func3_insns)
  //% last_insn = func3_insns.GetInstructionAtIndex(func3_insns.GetSize()-1)
  //% addr = last_insn.GetAddress()
  //% if "GNU" in self.name: addr.OffsetAddress(last_insn.GetByteSize())
  //% self.assertEqual(frame1.GetPCAddress(), addr)
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
