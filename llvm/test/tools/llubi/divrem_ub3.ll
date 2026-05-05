; RUN: sed 's/DIVREM/sdiv/g' %s | not llubi --verbose 2>&1 | FileCheck %s
; RUN: sed 's/DIVREM/srem/g' %s | not llubi --verbose 2>&1 | FileCheck %s

define void @main() {
  %res = DIVREM i8 -128, -1
  ret void
}
; CHECK: Entering function: main
; CHECK-NEXT: Stacktrace:
; CHECK-NEXT: #0 %res = {{sdiv|udiv|srem|urem}} i8 -128, -1 at @main
; CHECK-NEXT: Immediate UB detected: Signed division overflow.
; CHECK-NEXT: error: Execution of function 'main' failed.
