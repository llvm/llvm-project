; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare void @llvm.test.immarg.intrinsic.i32(i32 immarg)
declare void @llvm.test.immarg.intrinsic.v2i32(<2 x i32> immarg)
declare void @llvm.test.immarg.intrinsic.f32(float immarg)
declare void @llvm.test.immarg.intrinsic.v2f32(<2 x float> immarg)
declare void @llvm.test.immarg.intrinsic.2ai32([2 x i32] immarg)

@gv = global i32 undef, align 4

define void @call_llvm.test.immarg.intrinsic.i32(i32 %arg) {
; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 undef
; CHECK-NEXT: call void @llvm.test.immarg.intrinsic.i32(i32 undef)
  call void @llvm.test.immarg.intrinsic.i32(i32 undef)

; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 %arg
; CHECK-NEXT: call void @llvm.test.immarg.intrinsic.i32(i32 %arg)
  call void @llvm.test.immarg.intrinsic.i32(i32 %arg)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 ptrtoint (ptr @gv to i32)
  ; CHECK-NEXT: call void @llvm.test.immarg.intrinsic.i32(i32 ptrtoint (ptr @gv to i32))
  call void @llvm.test.immarg.intrinsic.i32(i32 ptrtoint (ptr @gv to i32))
  ret void
}

define void @call_llvm.test.immarg.intrinsic.f32() {
; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: float undef
; CHECK-NEXT: call void @llvm.test.immarg.intrinsic.f32(float undef)
  call void @llvm.test.immarg.intrinsic.f32(float undef)
  ret void
}

define void @call_llvm.test.immarg.intrinsic.v2i32() {
; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: <2 x i32> zeroinitializer
; CHECK-NEXT: call void @llvm.test.immarg.intrinsic.v2i32(<2 x i32> zeroinitializer)
  call void @llvm.test.immarg.intrinsic.v2i32(<2 x i32> zeroinitializer)

; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: <2 x i32> <i32 1, i32 2>
; CHECK-NEXT: call void @llvm.test.immarg.intrinsic.v2i32(<2 x i32> <i32 1, i32 2>)
  call void @llvm.test.immarg.intrinsic.v2i32(<2 x i32> <i32 1, i32 2>)

; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: <2 x i32> undef
; CHECK-NEXT: call void @llvm.test.immarg.intrinsic.v2i32(<2 x i32> undef)
  call void @llvm.test.immarg.intrinsic.v2i32(<2 x i32> undef)
  ret void
}

define void @call_llvm.test.immarg.intrinsic.v2f32() {
; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: <2 x float> zeroinitializer
; CHECK-NEXT: call void @llvm.test.immarg.intrinsic.v2f32(<2 x float> zeroinitializer)
  call void @llvm.test.immarg.intrinsic.v2f32(<2 x float> zeroinitializer)

; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: <2 x float> <float 1.000000e+00, float 2.000000e+00>
; CHECK-NEXT: call void @llvm.test.immarg.intrinsic.v2f32(<2 x float> <float 1.000000e+00, float 2.000000e+00>)
  call void @llvm.test.immarg.intrinsic.v2f32(<2 x float> <float 1.0, float 2.0>)
  ret void
}

define void @call_llvm.test.immarg.intrinsic.2ai32() {
; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: [2 x i32] zeroinitializer
; CHECK-NEXT: call void @llvm.test.immarg.intrinsic.2ai32([2 x i32] zeroinitializer)
  call void @llvm.test.immarg.intrinsic.2ai32([2 x i32] zeroinitializer)

; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: [2 x i32] [i32 1, i32 2]
; CHECK-NEXT: call void @llvm.test.immarg.intrinsic.2ai32([2 x i32] [i32 1, i32 2])
  call void @llvm.test.immarg.intrinsic.2ai32([2 x i32] [i32 1, i32 2])
  ret void
}

; CHECK: immarg attribute only applies to intrinsics
; CHECK-NEXT: ptr @not_an_intrinsic
declare void @not_an_intrinsic(i32 immarg)

declare void @llvm.test.intrinsic(i32)
declare void @func(i32)

define void @only_on_callsite() {
; CHECK: immarg attribute only applies to intrinsics
; CHECK-NEXT: call void @func(i32 immarg 0)
; CHECK-NEXT: immarg may not apply only to call sites
; CHECK-NEXT: i32 0
; CHECK-NEXT: call void @func(i32 immarg 0)
  call void @func(i32 immarg 0)

; CHECK: immarg may not apply only to call sites
; CHECK-NEXT: i32 0
; CHECK-NEXT: call void @llvm.test.intrinsic(i32 immarg 0)
  call void @llvm.test.intrinsic(i32 immarg 0)
  ret void
}

; CHECK: immarg attribute only applies to intrinsics
; CHECK: ptr @on_function_definition
define void @on_function_definition(i32 immarg %arg) {
  ret void
}
