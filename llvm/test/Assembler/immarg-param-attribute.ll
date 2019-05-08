; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK: declare void @llvm.test.immarg.intrinsic.i32(i32 immarg)
declare void @llvm.test.immarg.intrinsic.i32(i32 immarg)

; CHECK: declare void @llvm.test.immarg.intrinsic.f32(float immarg)
declare void @llvm.test.immarg.intrinsic.f32(float immarg)

; CHECK-LABEL: @call_llvm.test.immarg.intrinsic.i32(
define void @call_llvm.test.immarg.intrinsic.i32() {
  ; CHECK: call void @llvm.test.immarg.intrinsic.i32(i32 0)
  call void @llvm.test.immarg.intrinsic.i32(i32 0)

  ; CHECK: call void @llvm.test.immarg.intrinsic.i32(i32 0)
  call void @llvm.test.immarg.intrinsic.i32(i32 0)

  ; CHECK call void @llvm.test.immarg.intrinsic.i32(i32 1)
  call void @llvm.test.immarg.intrinsic.i32(i32 1)

  ; CHECK: call void @llvm.test.immarg.intrinsic.i32(i32 5)
  call void @llvm.test.immarg.intrinsic.i32(i32 add (i32 2, i32 3))

  ; CHECK: call void @llvm.test.immarg.intrinsic.i32(i32 0)
  call void @llvm.test.immarg.intrinsic.i32(i32 ptrtoint (i32* null to i32))
  ret void
}

; CHECK-LABEL: @call_llvm.test.immarg.intrinsic.f32(
define void @call_llvm.test.immarg.intrinsic.f32() {
  ; CHECK: call void @llvm.test.immarg.intrinsic.f32(float 1.000000e+00)
  call void @llvm.test.immarg.intrinsic.f32(float 1.0)
  ret void
}

define void @on_callsite_and_declaration() {
  ; CHECK: call void @llvm.test.immarg.intrinsic.i32(i32 immarg 0)
  call void @llvm.test.immarg.intrinsic.i32(i32 immarg 0)
  ret void
}
