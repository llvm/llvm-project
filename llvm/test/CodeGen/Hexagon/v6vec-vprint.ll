; RUN: llc -no-integrated-as -march=hexagon -mcpu=hexagonv60 -mattr=+hvxv60,hvx-length64b -disable-hexagon-shuffle=0 -O2 -enable-hexagon-vector-print < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -no-integrated-as -march=hexagon -mcpu=hexagonv60 -mattr=+hvxv60,hvx-length64b -disable-hexagon-shuffle=0 -O2 -enable-hexagon-vector-print -trace-hex-vector-stores-only < %s | FileCheck --check-prefix=VSTPRINT %s
;   generate .long XXXX which is a vector debug print instruction.
; CHECK: .long 0x1dffe0
; CHECK: .long 0x1dffe0
; CHECK: .long 0x1dffe0
; VSTPRINT: .long 0x1dffe0
; VSTPRINT-NOT: .long 0x1dffe0
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a:0-n16:32"
target triple = "hexagon"

; Function Attrs: nounwind
define void @do_vecs(ptr nocapture readonly %a, ptr nocapture readonly %b, ptr nocapture %c) #0 {
entry:
  %0 = load <16 x i32>, ptr %a, align 4, !tbaa !1
  %1 = load <16 x i32>, ptr %b, align 4, !tbaa !1
  %2 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %0, <16 x i32> %1)
  store <16 x i32> %2, ptr %c, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0}

!0 = !{!"QuIC LLVM Hexagon Clang version 7.x-pre-unknown"}
!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
