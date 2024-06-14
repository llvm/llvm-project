; RUN: llc < %s -mcpu=atom | FileCheck %s

; ModuleID = 'bugpoint-reduced-simplified.bc'
target triple = "x86_64-apple-darwin12.5.0"

define i32 @DoLayout() {
entry:
  %tmp1 = load i16, ptr undef, align 2
  %tmp17 = load i16, ptr null, align 2
  %tmp19 = load i16, ptr undef, align 2
  %shl = shl i16 %tmp19, 1
  %add55 = add i16 %tmp17, %tmp1
  %add57 = add i16 %add55, %shl
  %conv60 = zext i16 %add57 to i32
  %add61 = add nsw i32 %conv60, 0
  %conv63 = and i32 %add61, 65535
  ret i32 %conv63
; CHECK: addw
}
