; RUN: sed 's/OP1/store i32 0/g' %s | not llubi --verbose 2>&1 | FileCheck %s
; RUN: sed 's/OP1/%res = load i32/g' %s | not llubi --verbose 2>&1 | FileCheck %s

define void @main() {
  OP1, ptr poison
  ret void
}
; CHECK: Entering function: main
; CHECK-NEXT: Stacktrace:
; CHECK-NEXT: #0 {{store i32 0|%res = load i32}}, ptr poison, align 4 at @main
; CHECK-NEXT: Immediate UB detected: Invalid memory access with a poison pointer.
; CHECK-NEXT: error: Execution of function 'main' failed.
