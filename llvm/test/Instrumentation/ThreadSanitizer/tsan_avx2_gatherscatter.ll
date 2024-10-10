; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -S | FileCheck %s

; target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

attributes #0 = { sanitize_thread }
attributes #1 = { "target-features"="+avx2" }
attributes #2 = { "target-features"="+avx512f" }

declare void @llvm.masked.scatter.v4f64.v4p0(<4 x double>, <4 x ptr>, i32 immarg, <4 x i1>)
declare <4 x double> @llvm.masked.gather.v4f64.v4p0(<4 x ptr>, i32 immarg, <4 x i1>, <4 x double>)

define void @scatter_4_double_mask(<4 x double> %a, <4 x ptr> %p, <4 x i1> %m) #0 #1 {
entry:
  tail call void @llvm.masked.scatter.v4f64.v4p0(<4 x double> %a, <4 x ptr> %p, i32 8, <4 x i1> %m)
  ret void
}
; CHECK: define void @scatter_4_double_mask(
; CHECK: %1 = bitcast <4 x i1> %m to i4
; CHECK-NEXT: call void @__tsan_scatter_vector4(<4 x ptr> %p, i32 8, i4 %1)

define void @scatter_4_double(<4 x double> %a, <4 x ptr> %p) #0 #1  {
entry:
  tail call void @llvm.masked.scatter.v4f64.v4p0(<4 x double> %a, <4 x ptr> %p, i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>)
  ret void
}
; CHECK: define void @scatter_4_double(
; CHECK: call void @__tsan_scatter_vector4(<4 x ptr> %p, i32 8, i4 bitcast (<4 x i1> <i1 true, i1 true, i1 true, i1 true> to i4))

define void @gather_4_double(<4 x double> %a, <4 x ptr> %p) #0 #1 {
entry:
  tail call <4 x double> @llvm.masked.gather.v4f64.v4p0(<4 x ptr> %p, i32 8, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x double> %a)
  ret void
}
; CHECK: define void @gather_4_double(
; CHECK: call void @__tsan_gather_vector4(<4 x ptr> %p, i32 8, i4 bitcast (<4 x i1> <i1 true, i1 true, i1 true, i1 true> to i4))

define void @scatter_4_double_noavx(<4 x double> %a, <4 x ptr> %p, <4 x i1> %m) #0 {
entry:
  tail call void @llvm.masked.scatter.v4f64.v4p0(<4 x double> %a, <4 x ptr> %p, i32 8, <4 x i1> %m)
  ret void
}
; CHECK: define void @scatter_4_double_noavx(
; CHECK-NOT: call void @__tsan_scatter_vector

define void @scatter_4_double_avx512f(<4 x double> %a, <4 x ptr> %p, <4 x i1> %m) #0 #2 {
entry:
  tail call void @llvm.masked.scatter.v4f64.v4p0(<4 x double> %a, <4 x ptr> %p, i32 8, <4 x i1> %m)
  ret void
}
; CHECK: define void @scatter_4_double_avx512f(
; CHECK: %1 = bitcast <4 x i1> %m to i4
; CHECK-NEXT: call void @__tsan_scatter_vector4(<4 x ptr> %p, i32 8, i4 %1)
