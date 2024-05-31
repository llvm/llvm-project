; RUN: opt -S  -dxil-intrinsic-expansion  < %s | FileCheck %s 

; Make sure dxil operation function calls for abs are generated for int vectors.

; CHECK-LABEL: abs_i16Vec2
define noundef <2 x i16> @abs_i16Vec2(<2 x i16> noundef %a) #0 {
entry:
; CHECK: sub <2 x i16> zeroinitializer, %a
; CHECK: call <2 x i16> @llvm.smax.v2i16(<2 x i16> %a, <2 x i16> %{{.*}})
  %elt.abs = call <2 x i16> @llvm.abs.v2i16(<2 x i16> %a, i1 false)
  ret <2 x i16> %elt.abs
}

; CHECK-LABEL: abs_i32Vec3
define noundef <3 x i32> @abs_i32Vec3(<3 x i32> noundef %a) #0 {
entry:
; CHECK: sub <3 x i32> zeroinitializer, %a
; CHECK: call <3 x i32> @llvm.smax.v3i32(<3 x i32> %a, <3 x i32> %{{.*}})
  %elt.abs = call <3 x i32> @llvm.abs.v3i32(<3 x i32> %a, i1 false)
  ret <3 x i32> %elt.abs
}

; CHECK-LABEL: abs_i64Vec4
define noundef <4 x i64> @abs_i64Vec4(<4 x i64> noundef %a) #0 {
entry:
; CHECK: sub <4 x i64> zeroinitializer, %a
; CHECK: call <4 x i64> @llvm.smax.v4i64(<4 x i64> %a, <4 x i64> %{{.*}})
  %elt.abs = call <4 x i64> @llvm.abs.v4i64(<4 x i64> %a, i1 false)
  ret <4 x i64> %elt.abs
}

declare <2 x i16> @llvm.abs.v2i16(<2 x i16>, i1 immarg)
declare <3 x i32> @llvm.abs.v3i32(<3 x i32>, i1 immarg)
declare <4 x i64> @llvm.abs.v4i64(<4 x i64>, i1 immarg)
