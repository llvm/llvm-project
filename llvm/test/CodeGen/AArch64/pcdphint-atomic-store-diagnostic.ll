; RUN: split-file %s %t
; RUN: not llvm-as %t/bad-value-type.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID-SIG
; RUN: not llvm-as %t/bad-order-type.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID-SIG
; RUN: not llvm-as %t/bad-pointer-type.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID-SIG

; INVALID-SIG: error: invalid intrinsic signature

;--- bad-value-type.ll
; The intrinsic's value operand must be i64.
define void @bad_value_type(ptr %p, i32 %v) {
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i32 %v, i32 0, i32 0, i32 32)
  ret void
}

;--- bad-order-type.ll
; The order operand type must be i32.
define void @bad_order_type(ptr %p, i64 %v) {
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i64 0, i32 0, i32 64)
  ret void
}

;--- bad-pointer-type.ll
; The pointer operand must be a pointer type.
define void @bad_pointer_type(i64 %p, i64 %v) {
  call void @llvm.aarch64.stshh.atomic.store.p0(i64 %p, i64 %v, i32 0, i32 0, i32 64)
  ret void
}
