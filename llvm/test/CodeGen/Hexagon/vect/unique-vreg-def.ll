; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: hexagon

; This test was asserting because getVRegDef() was called on a register with
; multiple defs.
; Checks that the test does not assert and vsub is generated.
; CHECK: vsub

target triple = "hexagon"

@v = common dso_local local_unnamed_addr global <32 x i32> zeroinitializer, align 128

; Function Attrs: nounwind
define dso_local void @hvx_twoSum(<32 x i32>* nocapture noundef writeonly %s2lo) local_unnamed_addr #0 {
entry:
  %0 = load <32 x i32>, <32 x i32>* @v, align 128
  %call = tail call inreg <32 x i32> @MY_Vsf_equals_Vqf32(<32 x i32> noundef %0) #3
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vsub.sf.128B(<32 x i32> %call, <32 x i32> %call)
  store <32 x i32> %1, <32 x i32>* @v, align 128
  store <32 x i32> %1, <32 x i32>* %s2lo, align 128
  ret void
}

declare dso_local inreg <32 x i32> @MY_Vsf_equals_Vqf32(<32 x i32> noundef) local_unnamed_addr #1

; Function Attrs: nofree nosync nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsub.sf.128B(<32 x i32>, <32 x i32>) #2

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv73" "target-features"="+hvx-length128b,+hvxv73,+v73,-long-calls" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv73" "target-features"="+hvx-length128b,+hvxv73,+v73,-long-calls" }
attributes #2 = { nofree nosync nounwind readnone }
attributes #3 = { nounwind }
