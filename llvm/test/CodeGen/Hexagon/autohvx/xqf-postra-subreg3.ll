; Test passes if there is no mismatch on a convert instruction
;
; UNSUPPORTED: asserts

; REQUIRES: asserts
; RUN: llc -O2 -mtriple=hexagon -mattr=+hvxv81,+hvx-length128B \
; RUN: -enable-xqf-gen=true -hexagon-qfloat-mode=lossy \
; RUN: -debug-only=handle-qfp -enable-postra-xqf-check < %s 2>&1 -o - | FileCheck %s

; CHECK: Mismatch: qf32 type used as sf at operand
; CHECK-NOT:  Def:  renamable $v{{[0-9]+}} = V6_vconv_qf32_sf renamable

declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #0

define tailcc void @widget(ptr %arg, ptr %arg1, i1 %arg2, i1 %arg3, <32 x i32> %arg4) {
bb:
  br label %bb5

bb5:                                              ; preds = %bb7, %bb
  br i1 %arg2, label %bb6, label %bb7

bb6:                                              ; preds = %bb5
  %load = load <32 x i32>, ptr %arg, align 128
  br label %bb7

bb7:                                              ; preds = %bb6, %bb5
  %phi = phi <32 x i32> [ %load, %bb6 ], [ zeroinitializer, %bb5 ]
  %call = tail call <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.acc.128B(<64 x i32> zeroinitializer, <32 x i32> %phi, <32 x i32> zeroinitializer)
  tail call void (i32, i32, ptr, ...) %arg(i32 0, i32 0, ptr null, ptr null, i32 0, ptr null, ptr null)
  %call8 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.acc.128B(<64 x i32> %call, <32 x i32> zeroinitializer, <32 x i32> zeroinitializer)
  %call9 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.acc.128B(<64 x i32> %call8, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <32 x i32> zeroinitializer)
  %call10 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %call9)
  %call11 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32> zeroinitializer, <32 x i32> %call10)
  store <32 x i32> %call11, ptr %arg1, align 128
  br i1 %arg3, label %bb5, label %bb12

bb12:                                             ; preds = %bb12, %bb7
  store <32 x i32> %arg4, ptr %arg, align 128
  br label %bb12
}

declare <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.acc.128B(<64 x i32>, <32 x i32>, <32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32>, <32 x i32>) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
