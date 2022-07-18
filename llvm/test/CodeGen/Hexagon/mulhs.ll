; RUN: llc -march=hexagon < %s | FileCheck %s
; RUN: llc -march=hexagon -early-live-intervals -verify-machineinstrs < %s | FileCheck %s

; CHECK: mpy
; CHECK-NOT: call

target triple = "hexagon"

; Function Attrs: nounwind
define i32 @fred(i64 %x, i64 %y, i64* nocapture %z) #0 {
entry:
  %0 = tail call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %x, i64 %y)
  %1 = extractvalue { i64, i1 } %0, 1
  %2 = extractvalue { i64, i1 } %0, 0
  store i64 %2, i64* %z, align 8
  %conv = zext i1 %1 to i32
  ret i32 %conv
}

; Function Attrs: nounwind readnone
declare { i64, i1 } @llvm.smul.with.overflow.i64(i64, i64) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
