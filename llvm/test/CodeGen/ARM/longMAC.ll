; RUN: llc < %s -march=arm | FileCheck %s
; Check generated signed and unsigned multiply accumulate long.

define i64 @MACLongTest1(i32 %a, i32 %b, i64 %c) {
;CHECK: MACLongTest1:
;CHECK: umlal
  %conv = zext i32 %a to i64
  %conv1 = zext i32 %b to i64
  %mul = mul i64 %conv1, %conv
  %add = add i64 %mul, %c
  ret i64 %add
}

define i64 @MACLongTest2(i32 %a, i32 %b, i64 %c)  {
;CHECK: MACLongTest2:
;CHECK: smlal
  %conv = sext i32 %a to i64
  %conv1 = sext i32 %b to i64
  %mul = mul nsw i64 %conv1, %conv
  %add = add nsw i64 %mul, %c
  ret i64 %add
}

define i64 @MACLongTest3(i32 %a, i32 %b, i32 %c) {
;CHECK: MACLongTest3:
;CHECK: umlal
  %conv = zext i32 %b to i64
  %conv1 = zext i32 %a to i64
  %mul = mul i64 %conv, %conv1
  %conv2 = zext i32 %c to i64
  %add = add i64 %mul, %conv2
  ret i64 %add
}

define i64 @MACLongTest4(i32 %a, i32 %b, i32 %c) {
;CHECK: MACLongTest4:
;CHECK: smlal
  %conv = sext i32 %b to i64
  %conv1 = sext i32 %a to i64
  %mul = mul nsw i64 %conv, %conv1
  %conv2 = sext i32 %c to i64
  %add = add nsw i64 %mul, %conv2
  ret i64 %add
}
