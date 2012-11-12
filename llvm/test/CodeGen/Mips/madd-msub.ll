; RUN: llc -march=mips < %s | FileCheck %s

; CHECK: madd 
define i64 @madd1(i32 %a, i32 %b, i32 %c) nounwind readnone {
entry:
  %conv = sext i32 %a to i64
  %conv2 = sext i32 %b to i64
  %mul = mul nsw i64 %conv2, %conv
  %conv4 = sext i32 %c to i64
  %add = add nsw i64 %mul, %conv4
  ret i64 %add
}

; CHECK: maddu
define i64 @madd2(i32 %a, i32 %b, i32 %c) nounwind readnone {
entry:
  %conv = zext i32 %a to i64
  %conv2 = zext i32 %b to i64
  %mul = mul nsw i64 %conv2, %conv
  %conv4 = zext i32 %c to i64
  %add = add nsw i64 %mul, %conv4
  ret i64 %add
}

; CHECK: madd
define i64 @madd3(i32 %a, i32 %b, i64 %c) nounwind readnone {
entry:
  %conv = sext i32 %a to i64
  %conv2 = sext i32 %b to i64
  %mul = mul nsw i64 %conv2, %conv
  %add = add nsw i64 %mul, %c
  ret i64 %add
}

; CHECK: msub
define i64 @msub1(i32 %a, i32 %b, i32 %c) nounwind readnone {
entry:
  %conv = sext i32 %c to i64
  %conv2 = sext i32 %a to i64
  %conv4 = sext i32 %b to i64
  %mul = mul nsw i64 %conv4, %conv2
  %sub = sub nsw i64 %conv, %mul
  ret i64 %sub
}

; CHECK: msubu 
define i64 @msub2(i32 %a, i32 %b, i32 %c) nounwind readnone {
entry:
  %conv = zext i32 %c to i64
  %conv2 = zext i32 %a to i64
  %conv4 = zext i32 %b to i64
  %mul = mul nsw i64 %conv4, %conv2
  %sub = sub nsw i64 %conv, %mul
  ret i64 %sub
}

; CHECK: msub 
define i64 @msub3(i32 %a, i32 %b, i64 %c) nounwind readnone {
entry:
  %conv = sext i32 %a to i64
  %conv3 = sext i32 %b to i64
  %mul = mul nsw i64 %conv3, %conv
  %sub = sub nsw i64 %c, %mul
  ret i64 %sub
}
