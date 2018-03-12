; RUN: llc -march=hexagon < %s
; REQUIRES: asserts
; The test case validates the fact that if the modifier register value "-268430336"
; is passed as target constant, then the compiler must not assert.
; This test also validates that the VLIW Packetizer does not bail out the compilation
; with "Unknown .new type" when attempting to validate if the circular store can be
; converted to a new value store.

target triple = "hexagon"

; Function Attrs: nounwind
define zeroext i8 @f0(i8* %a0) local_unnamed_addr #0 {
b0:
  %v0 = tail call { i32, i8* } @llvm.hexagon.L2.loadrub.pcr(i8* %a0, i32 -268430336, i8* %a0)
  %v1 = extractvalue { i32, i8* } %v0, 0
  %v2 = trunc i32 %v1 to i8
  ret i8 %v2
}

; Function Attrs: argmemonly nounwind
declare { i32, i8* } @llvm.hexagon.L2.loadrub.pcr(i8*, i32, i8* nocapture) #1

; Function Attrs: nounwind
define void @f1(i8* %a0, i8 zeroext %a1) local_unnamed_addr #0 {
b0:
  %v0 = zext i8 %a1 to i32
  %v1 = tail call i8* @llvm.hexagon.S2.storerb.pcr(i8* %a0, i32 -268430336, i32 %v0, i8* %a0)
  ret void
}

; Function Attrs: argmemonly nounwind
declare i8* @llvm.hexagon.S2.storerb.pcr(i8*, i32, i32, i8* nocapture) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { argmemonly nounwind }
