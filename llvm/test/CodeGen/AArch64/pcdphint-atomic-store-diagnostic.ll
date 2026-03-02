; RUN: split-file %s %t
; RUN: not llvm-as %t/bad-value-type.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID-SIG
; RUN: not llvm-as %t/bad-order-type.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID-SIG
; RUN: not llvm-as %t/bad-pointer-type.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID-SIG
; RUN: not llvm-as %t/bad-order-value.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID-ORDER
; RUN: not llvm-as %t/bad-policy-value.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID-POLICY
; RUN: not llvm-as %t/bad-size-value.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID-SIZE

; INVALID-SIG: error: invalid intrinsic signature
; INVALID-ORDER: order argument to llvm.aarch64.stshh.atomic.store must be 0, 3 or 5
; INVALID-POLICY: policy argument to llvm.aarch64.stshh.atomic.store must be 0 or 1
; INVALID-SIZE: size argument to llvm.aarch64.stshh.atomic.store must be 8, 16, 32 or 64

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

;--- bad-order-value.ll
; The order operand value must be one of 0, 3, or 5.
define void @bad_order_value(ptr %p, i64 %v) {
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 1, i32 0, i32 64)
  ret void
}

;--- bad-policy-value.ll
; The policy operand value must be 0 (keep) or 1 (strm).
define void @bad_policy_value(ptr %p, i64 %v) {
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 0, i32 2, i32 64)
  ret void
}

;--- bad-size-value.ll
; The size operand value must be one of 8, 16, 32, or 64.
define void @bad_size_value(ptr %p, i64 %v) {
  call void @llvm.aarch64.stshh.atomic.store.p0(ptr %p, i64 %v, i32 0, i32 0, i32 0)
  ret void
}
