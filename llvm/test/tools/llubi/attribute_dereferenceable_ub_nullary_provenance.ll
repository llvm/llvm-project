; RUN: not llubi --verbose < %s 2>&1 | FileCheck %s
; RUN: sed 's/dereferenceable/dereferenceable_or_null/g' %s | not llubi --verbose 2>&1 | FileCheck %s

define void @callee(ptr dereferenceable(4) %x) {
  ret void
}

define void @main() {
  %ptr_storage = alloca i64
  %p = load ptr, ptr %ptr_storage
  call void @callee(ptr %p)
  ret void
}
; CHECK: Entering function: main
; CHECK-NEXT:   %ptr_storage = alloca i64, align 8 => ptr 0x8 [ptr_storage]
; CHECK-NEXT:   %p = load ptr, ptr %ptr_storage, align 8 => ptr 0xE82FEEACEEB98B3E [dangling]
; CHECK-NEXT: Immediate UB detected: The value violates dereferenceable{{(_or_null)?}} attribute.
; CHECK-NEXT: error: Execution of function 'main' failed.
