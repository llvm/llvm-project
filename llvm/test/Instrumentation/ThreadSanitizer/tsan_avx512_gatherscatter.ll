; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -S | FileCheck %s

; target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

attributes #0 = { sanitize_thread }
attributes #1 = { "target-features"="+avx2" }
attributes #2 = { "target-features"="+avx512f" }

declare void @llvm.masked.scatter.v8f64.v8p0(<8 x double>, <8 x ptr>, i32 immarg, <8 x i1>)
declare void @llvm.masked.scatter.v8f32.v8p0(<8 x float>, <8 x ptr>, i32 immarg, <8 x i1>)
declare <8 x double> @llvm.masked.gather.v8f64.v8p0(<8 x ptr>, i32 immarg, <8 x i1>, <8 x double>)
declare <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr>, i32 immarg, <8 x i1>, <8 x float>)

define void @scatter_8_double_mask(<8 x double> %a, <8 x ptr> %p, <8 x i1> %m) #0 #2 {
entry:
  tail call void @llvm.masked.scatter.v8f64.v8p0(<8 x double> %a, <8 x ptr> %p, i32 8, <8 x i1> %m)
  ret void
}
; CHECK: define void @scatter_8_double_mask(
; CHECK: %1 = bitcast <8 x i1> %m to i8
; CHECK-NEXT: call void @__tsan_scatter_vector8(<8 x ptr> %p, i32 8, i8 %1)

define void @scatter_8_float_mask(<8 x float> %a, <8 x ptr> %p, <8 x i1> %m) #0 #2 {
entry:
  tail call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %a, <8 x ptr> %p, i32 4, <8 x i1> %m)
  ret void
}
; CHECK: define void @scatter_8_float_mask(
; CHECK: %1 = bitcast <8 x i1> %m to i8
; CHECK-NEXT: call void @__tsan_scatter_vector8(<8 x ptr> %p, i32 4, i8 %1)

define void @scatter_8_double(<8 x double> %a, <8 x ptr> %p) #0 #2 {
entry:
  tail call void @llvm.masked.scatter.v8f64.v8p0(<8 x double> %a, <8 x ptr> %p, i32 8, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  ret void
}
; CHECK: define void @scatter_8_double(
; CHECK: call void @__tsan_scatter_vector8(<8 x ptr> %p, i32 8, i8 bitcast (<8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true> to i8))

define void @gather_8_double_mask(<8 x double> %a, <8 x ptr> %p, <8 x i1> %m) #0 #2 {
entry:
  tail call <8 x double> @llvm.masked.gather.v8f64.v8p0(<8 x ptr> %p, i32 8, <8 x i1> %m, <8 x double> %a)
  ret void
}
; CHECK: define void @gather_8_double_mask(
; CHECK: %1 = bitcast <8 x i1> %m to i8
; CHECK-NEXT: call void @__tsan_gather_vector8(<8 x ptr> %p, i32 8, i8 %1)

define void @gather_8_float_mask(<8 x float> %a, <8 x ptr> %p, <8 x i1> %m) #0 #2 {
entry:
  tail call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %p, i32 4, <8 x i1> %m, <8 x float> %a)
  ret void
}
; CHECK: define void @gather_8_float_mask(
; CHECK: %1 = bitcast <8 x i1> %m to i8
; CHECK-NEXT: call void @__tsan_gather_vector8(<8 x ptr> %p, i32 4, i8 %1)

define void @gather_8_double(<8 x double> %a, <8 x ptr> %p) #0 #2 {
entry:
  tail call <8 x double> @llvm.masked.gather.v8f64.v8p0(<8 x ptr> %p, i32 8, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x double> %a)
  ret void
}
; CHECK: define void @gather_8_double(
; CHECK: call void @__tsan_gather_vector8(<8 x ptr> %p, i32 8, i8 bitcast (<8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true> to i8))

define void @gather_8_float_avx2(<8 x float> %a, <8 x ptr> %p, <8 x i1> %m) #0 #1 {
entry:
  tail call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %p, i32 4, <8 x i1> %m, <8 x float> %a)
  ret void
}
; CHECK: define void @gather_8_float_avx2(
; CHECK-NOT: call void @__tsan_gather_vector

define void @gather_8_float_noavx(<8 x float> %a, <8 x ptr> %p, <8 x i1> %m) #0 {
entry:
  tail call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %p, i32 4, <8 x i1> %m, <8 x float> %a)
  ret void
}
; CHECK: define void @gather_8_float_noavx(
; CHECK-NOT: call void @__tsan_gather_vector
