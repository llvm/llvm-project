; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that intrinsic int_hexagon_Y2_dcfetch is mapped to Y2_dcfetchbo
; (not Y2_dcfetch).

; CHECK: dcfetch(r0+#0)

target triple = "hexagon"

define void @fred(i8* %a0) #0 {
  call void @llvm.hexagon.Y2.dcfetch(i8* %a0)
  ret void
}

declare void @llvm.hexagon.Y2.dcfetch(i8*) #0

attributes #0 = { nounwind }

