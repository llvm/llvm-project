; RUN: not llubi --verbose < %s 2>&1 | FileCheck %s
; RUN: sed 's/dereferenceable/dereferenceable_or_null/g' %s | not llubi --verbose 2>&1 | FileCheck %s

define void @callee(ptr dereferenceable(16) %x) {
  ret void
}

define void @main() {
  %alloc = alloca i64
  call void @callee(ptr %alloc)
  ret void
}
; CHECK: Entering function: main
; CHECK-NEXT:   %alloc = alloca i64, align 8 => ptr 0x8 [alloc]
; CHECK-NEXT: Stacktrace:
; CHECK-NEXT: #0   call void @callee(ptr %alloc) at @main <stdin>:10
; CHECK-NEXT: Immediate UB detected: The value ptr 0x8 [alloc] violates dereferenceable{{(_or_null)?}}(16) attribute.
; CHECK-NEXT: error: Execution of function 'main' failed.
