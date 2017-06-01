; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we don't crash.
; CHECK: vshuff

target triple = "hexagon"

define void @hex_interleaved.s0.__outermost() local_unnamed_addr #0 {
entry:
  %0 = icmp eq i32 undef, 0
  %sel2 = select i1 %0, <32 x i16> undef, <32 x i16> zeroinitializer
  %1 = bitcast <32 x i16> %sel2 to <16 x i32>
  %2 = tail call <16 x i32> @llvm.hexagon.V6.vshuffh(<16 x i32> %1)
  store <16 x i32> %2, <16 x i32>* undef, align 2
  unreachable
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshuffh(<16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx" }
attributes #1 = { nounwind readnone }
