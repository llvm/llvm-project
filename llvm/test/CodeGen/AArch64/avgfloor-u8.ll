; RUN: llc < %s -march=arm64 -mcpu=apple-m1 | FileCheck %s

; CHECK-LABEL: avg:
; CHECK:       add
; CHECK:       lsr
; CHECK:       ret

define zeroext i8 @avg(i8 noundef zeroext %a, i8 noundef zeroext %b) {
entry:
  %conv = zext i8 %a to i16
  %conv1 = zext i8 %b to i16
  %add = add nuw nsw i16 %conv1, %conv
  %div3 = lshr i16 %add, 1
  %conv2 = trunc nuw i16 %div3 to i8
  ret i8 %conv2
}
