; RUN: not opt -passes=verify -disable-output < %s 2>&1 | FileCheck %s

define void @main() {
; CHECK: fptosi.sat/fptoui.sat argument and result types must both be either vector or scalar types
; CHECK-NEXT: %1 = call i32 @llvm.fptosi.sat.i32.v1f32(<1 x float> zeroinitializer)
  call i32 @llvm.fptosi.sat.i32.v1f32(<1 x float> zeroinitializer)
; CHECK: return type and arguments must have the same number of elements
; CHECK-NEXT: %2 = call <2 x i32> @llvm.fptosi.sat.v2i32.v1f32(<1 x float> zeroinitializer)
  call <2 x i32> @llvm.fptosi.sat.v2i32.v1f32(<1 x float> zeroinitializer)
; CHECK: fptosi.sat/fptoui.sat argument and result types must both be either vector or scalar types
; CHECK-NEXT: %3 = call i32 @llvm.fptoui.sat.i32.v1f32(<1 x float> zeroinitializer)
  call i32 @llvm.fptoui.sat.i32.v1f32(<1 x float> zeroinitializer)
; CHECK: return type and arguments must have the same number of elements
; CHECK-NEXT: %4 = call <2 x i32> @llvm.fptoui.sat.v2i32.v1f32(<1 x float> zeroinitializer)
  call <2 x i32> @llvm.fptoui.sat.v2i32.v1f32(<1 x float> zeroinitializer)
  ret void
}
