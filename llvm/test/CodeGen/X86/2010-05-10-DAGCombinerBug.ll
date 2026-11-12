; RUN: llc < %s -mtriple=i386-apple-darwin10
; PR7018
; rdar://7939869

define i32 @CXB30130(i32 %num1, ptr nocapture %num2, ptr nocapture %num3, ptr nocapture %num4) nounwind ssp {
entry:
  %0 = load i16, ptr %num2, align 2                   ; <i16> [#uses=2]
  %1 = mul nsw i16 %0, %0                         ; <i16> [#uses=1]
  store i16 %1, ptr %num2, align 2
  ret i32 undef
}
