; RUN: llc -O2 -mtriple=hexagon < %s | FileCheck %s

; Hexagon's vsplatb/vsplath only consider the lower 8/16 bits of the source
; register.  Any extension of the source is not necessary.

; CHECK-NOT: zxtb
; CHECK-NOT: zxth

target triple = "hexagon"

; Function Attrs: nounwind readnone
define i64 @f0(i64 %a0) #0 {
b0:
  %v0 = trunc i64 %a0 to i32
  %v1 = and i32 %v0, 65535
  %v2 = tail call i64 @llvm.hexagon.S2.vsplatrh(i32 %v1)
  %v3 = and i32 %v0, 255
  %v4 = tail call i32 @llvm.hexagon.S2.vsplatrb(i32 %v3)
  %v5 = sext i32 %v4 to i64
  %v6 = add nsw i64 %v5, %v2
  ret i64 %v6
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.vsplatrh(i32) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.vsplatrb(i32) #0

attributes #0 = { nounwind readnone }
