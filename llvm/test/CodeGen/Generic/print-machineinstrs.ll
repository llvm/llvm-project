; RUN: llc < %s -O3 -debug-pass=Structure -print-machineinstrs=branch-folder -o /dev/null 2>&1 | FileCheck %s
; RUN: llc < %s -O3 -debug-pass=Structure -print-machineinstrs -o /dev/null 2>&1 | FileCheck %s
; RUN: llc < %s -O3 -debug-pass=Structure -print-machineinstrs= -o /dev/null 2>&1 | FileCheck %s

define i64 @foo(i64 %a, i64 %b) nounwind {
; CHECK: -branch-folder -print-machineinstrs
; CHECK: Control Flow Optimizer
; CHECK-NEXT: MachineFunction Printer
; CHECK: Machine code for function foo:
  %c = add i64 %a, %b
  %d = trunc i64 %c to i32
  %e = zext i32 %d to i64
  ret i64 %e
}
