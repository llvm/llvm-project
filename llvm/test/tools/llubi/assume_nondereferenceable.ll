; RUN: not llubi --verbose < %s 2>&1 | FileCheck %s
; RUN: sed 's/dereferenceable/dereferenceable_or_null/g' %s | not llubi --verbose 2>&1 | FileCheck %s

define void @main() {
  %alloc = alloca i32
  call void @llvm.assume(i1 true) ["dereferenceable"(ptr %alloc, i32 2048)]
  ret void
}
; CHECK: Entering function: main
; CHECK-NEXT:   %alloc = alloca i32, align 4 => ptr 0x8 [alloc]
; CHECK-NEXT: Stacktrace:
; CHECK-NEXT: #0   call void @llvm.assume(i1 true) [ "dereferenceable{{(_or_null)?}}"(ptr %alloc, i32 2048) ] at @main
; CHECK-NEXT: Immediate UB detected: The pointer ptr 0x8 [alloc] violates dereferenceable{{(_or_null)?}}(2048) assumption.
; CHECK-NEXT: error: Execution of function 'main' failed.
