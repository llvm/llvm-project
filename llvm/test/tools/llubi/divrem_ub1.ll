; RUN: sed 's/DIVREM/sdiv/g' %s | not llubi --verbose 2>&1 | FileCheck %s
; RUN: sed 's/DIVREM/udiv/g' %s | not llubi --verbose 2>&1 | FileCheck %s
; RUN: sed 's/DIVREM/srem/g' %s | not llubi --verbose 2>&1 | FileCheck %s
; RUN: sed 's/DIVREM/urem/g' %s | not llubi --verbose 2>&1 | FileCheck %s

define void @main() {
  %res = DIVREM <2 x i32> splat(i32 10), zeroinitializer
  ret void
}
; CHECK: Entering function: main
; CHECK-NEXT: Immediate UB detected: Division by zero.
; CHECK-NEXT: error: Execution of function 'main' failed.
