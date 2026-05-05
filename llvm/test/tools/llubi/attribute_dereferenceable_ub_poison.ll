; RUN: not llubi --verbose < %s 2>&1 | FileCheck %s
; RUN: sed 's/dereferenceable/dereferenceable_or_null/g' %s | not llubi --verbose 2>&1 | FileCheck %s

define void @callee(ptr dereferenceable(4) %x) {
  ret void
}

define void @main() {
  call void @callee(ptr poison)
  ret void
}
; CHECK: Entering function: main
; CHECK-NEXT: Stacktrace:
; CHECK-NEXT: #0   call void @callee(ptr poison) at @main
; CHECK-NEXT: Immediate UB detected: The value poison violates dereferenceable{{(_or_null)?}}(4) attribute.
; CHECK-NEXT: error: Execution of function 'main' failed.
