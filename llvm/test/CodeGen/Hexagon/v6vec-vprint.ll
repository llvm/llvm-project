; RUN: llc -march=hexagon -mcpu=hexagonv60 -enable-hexagon-hvx -disable-hexagon-shuffle=0 -O2 -enable-hexagon-vector-print < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -march=hexagon -mcpu=hexagonv60 -enable-hexagon-hvx -disable-hexagon-shuffle=0 -O2 -enable-hexagon-vector-print -trace-hex-vector-stores-only < %s | FileCheck --check-prefix=VSTPRINT %s
;   generate .long XXXX which is a vector debug print instruction.
; CHECK: .long 0x1dffe0
; CHECK: .long 0x1dffe0
; CHECK: .long 0x1dffe0
; VSTPRINT: .long 0x1dffe0
; VSTPRINT-NOT: .long 0x1dffe0
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a:0-n16:32"
target triple = "hexagon"

; Function Attrs: nounwind
define void @do_vecs(i8* nocapture readonly %a, i8* nocapture readonly %b, i8* nocapture %c) #0 {
entry:
  %0 = bitcast i8* %a to <16 x i32>*
  %1 = load <16 x i32>, <16 x i32>* %0, align 4, !tbaa !1
  %2 = bitcast i8* %b to <16 x i32>*
  %3 = load <16 x i32>, <16 x i32>* %2, align 4, !tbaa !1
  %4 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %1, <16 x i32> %3)
  %5 = bitcast i8* %c to <16 x i32>*
  store <16 x i32> %4, <16 x i32>* %5, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0}

!0 = !{!"QuIC LLVM Hexagon Clang version 7.x-pre-unknown"}
!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
