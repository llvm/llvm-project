; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "p1:64:64:64:32"

declare float @llvm.ptrmask.f32.i64(float, i64)
declare ptr @llvm.ptrmask.p0.i32(ptr, i32)
declare ptr addrspace(1) @llvm.ptrmask.p1.i64(ptr addrspace(1), i64)

; CHECK: llvm.ptrmask intrinsic first argument must be pointer or vector of pointers
; CHECK-NEXT:  %1 = call float @llvm.ptrmask.f32.i64(float 0.000000e+00, i64 0)
define void @not_ptr() {
  call float @llvm.ptrmask.f32.i64(float 0.0, i64 0)
  ret void
}

; CHECK: llvm.ptrmask intrinsic second argument bitwidth must match pointer index type size of first argument
; CHECK-NEXT:  %1 = call ptr @llvm.ptrmask.p0.i32(ptr null, i32 0)
define void @wrong_size_1() {
  call ptr @llvm.ptrmask.p0.i32(ptr null, i32 0)
  ret void
}

; CHECK: llvm.ptrmask intrinsic second argument bitwidth must match pointer index type size of first argument
; CHECK-NEXT:  %1 = call ptr addrspace(1) @llvm.ptrmask.p1.i64(ptr addrspace(1) null, i64 0)
define void @wrong_size_2() {
  call ptr addrspace(1) @llvm.ptrmask.p1.i64(ptr addrspace(1) null, i64 0)
  ret void
}
