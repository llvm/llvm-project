; RUN: sed 's/OP1/store i32 0/g' %s | not llubi --verbose 2>&1 | FileCheck %s
; RUN: sed 's/OP1/%res = load i32/g' %s | not llubi --verbose 2>&1 | FileCheck %s

define void @main() {
  OP1, ptr null
  ret void
}
; CHECK: Entering function: main
; CHECK-NEXT: Immediate UB detected: Invalid memory access via a pointer with nullary provenance.
; CHECK-NEXT: error: Execution of function 'main' failed.
