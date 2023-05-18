; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we don't crash.
; CHECK: vshuff

target triple = "hexagon"

define void @f0(ptr %a0) #0 {
entry:
  %v0 = icmp eq i32 undef, 0
  %v1 = select i1 %v0, <32 x i16> undef, <32 x i16> zeroinitializer
  %v2 = bitcast <32 x i16> %v1 to <16 x i32>
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vshuffh(<16 x i32> %v2)
  store <16 x i32> %v3, ptr %a0, align 2
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshuffh(<16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
