; RUN: llc < %s
; REQUIRES: asserts
target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon"

; Function Attrs: nounwind readnone
define i64 @foo() #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S2.vsplatrb(i32 255)
  %conv = zext i32 %0 to i64
  %shl = shl nuw i64 %conv, 32
  %or = or i64 %shl, %conv
  ret i64 %or
}

declare i32 @llvm.hexagon.S2.vsplatrb(i32) #0

attributes #0 = { nounwind readnone }
