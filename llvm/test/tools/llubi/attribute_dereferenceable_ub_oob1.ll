; RUN: not llubi --verbose < %s 2>&1 | FileCheck %s
; RUN: sed 's/dereferenceable/dereferenceable_or_null/g' %s | not llubi --verbose 2>&1 | FileCheck %s

define void @callee(ptr dereferenceable(8) %x) {
  ret void
}

define void @main() {
  %alloc = alloca i32
  call void @callee(ptr %alloc)
  ret void
}
; CHECK: Entering function: main
; CHECK-NEXT:   %alloc = alloca i32, align 4 => ptr 0x8 [alloc]
; CHECK-NEXT: Stacktrace:
; CHECK-NEXT: #0   call void @callee(ptr %alloc) at @main
; CHECK-NEXT: Immediate UB detected: The value ptr 0x8 [alloc] violates dereferenceable{{(_or_null)?}}(8) attribute.
; CHECK-NEXT: error: Execution of function 'main' failed.
