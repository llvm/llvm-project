; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s
define i32 @f(i32 %a, i32 %b, i32 %c) nounwind readnone ssp {
entry:
; CHECK-NOT: smmls
  %conv4 = zext i32 %a to i64
  %conv1 = sext i32 %b to i64
  %conv2 = sext i32 %c to i64
  %mul = mul nsw i64 %conv2, %conv1
  %shr5 = lshr i64 %mul, 32
  %sub = sub nsw i64 %conv4, %shr5
  %conv3 = trunc i64 %sub to i32
  ret i32 %conv3
}
