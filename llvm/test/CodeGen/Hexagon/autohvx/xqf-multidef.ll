; This is a unique case where an xqf use has two definitions. One def comes from
; a sf type, but another comes from a qf generating instruction.

; REQUIRES: asserts
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=ieee -mattr=+hvxv79,+hvx-length128B -debug-only=handle-qfp \
; RUN: 2>&1 < %s -o /dev/null | FileCheck %s
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv81 -force-hvx-float -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=ieee -mattr=+hvxv81,+hvx-length128B -debug-only=handle-qfp \
; RUN: 2>&1 < %s -o /dev/null | FileCheck %s

; CHECK: Instruction:   renamable [[V14:\$v[0-9]+]] = V6_vmpy_qf32 killed renamable [[V4:\$v[0-9]+]], killed renamable [[V5:\$v[0-9]+]]
; CHECK-NEXT: Property: 0 ,1
; CHECK: Processing:   renamable [[V14]] = V6_vmpy_qf32 killed renamable [[V4]], killed renamable [[V5]]
; CHECK: Inserting new instruction before:   [[V4]] = V6_vconv_sf_qf32 killed renamable [[V4]]
; CHECK: Inserting new instruction:   [[V14]] = V6_vmpy_qf32_sf killed renamable [[V4]], killed renamable [[V5]]

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define i32 @qhmath_hvx_sin_af(ptr %input, i32 %0) {
entry:
  br label %for.body12

for.cond.loopexit:                                ; preds = %for.body12
  ret i32 0

for.body12:                                       ; preds = %for.body12, %entry
  %j.0104 = phi i32 [ 0, %entry ], [ %inc, %for.body12 ]
  %1 = load <32 x i32>, ptr null, align 128
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.sf.128B(<32 x i32> zeroinitializer, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> zeroinitializer, <32 x i32> %1)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32> %3, <32 x i32> %2)
  store <32 x i32> %4, ptr %input, align 4
  %inc = add i32 %j.0104, 1
  %exitcond.not = icmp eq i32 %j.0104, %0
  br i1 %exitcond.not, label %for.cond.loopexit, label %for.body12
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vmpy.qf32.sf.128B(<32 x i32>, <32 x i32>) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(<32 x i32>, <32 x i32>) #0

; uselistorder directives
uselistorder ptr @llvm.hexagon.V6.vmpy.qf32.128B, { 1, 0 }

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
