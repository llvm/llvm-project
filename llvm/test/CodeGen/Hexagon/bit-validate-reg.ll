; RUN: llc -mtriple=hexagon -hexbit-extract=0 < %s | FileCheck %s

; Make sure we don't generate zxtb to transfer a predicate register into
; a general purpose register.

; CHECK: r0 = p0
; CHECK-NOT: zxtb(p
; CHECK-NOT: and(p
; CHECK-NOT: extract(p
; CHECK-NOT: extractu(p

target triple = "hexagon"

; Function Attrs: nounwind
define i32 @fred() local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.C4.and.and(i32 undef, i32 undef, i32 undef)
  ret i32 %0
}

declare i32 @llvm.hexagon.C4.and.and(i32, i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
attributes #1 = { nounwind readnone }
