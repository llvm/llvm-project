; RUN: llc -mcpu=core-avx2 -march=x86-64 < %s | FileCheck %s

define i128 @f1(i64 %a, i64 %b) {
  %x = zext i64 %a to i128
  %y = zext i64 %b to i128
  %r = mul i128 %x, %y
; CHECK: f1
; CHECK: mulxq
; CHECK: ret
  ret i128 %r
}

define i128 @f2(i64 %a, i64* %p) {
  %b = load i64* %p
  %x = zext i64 %a to i128
  %y = zext i64 %b to i128
  %r = mul i128 %x, %y
; CHECK: f2
; CHECK: mulxq ({{.+}}), %{{.+}}, %{{.+}}
; CHECK: ret
  ret i128 %r
}
