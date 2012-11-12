; RUN: llc -march=mips64el -mcpu=mips64 < %s | FileCheck %s

define i64 @m0(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: dmult
; CHECK: mflo
  %mul = mul i64 %a1, %a0
  ret i64 %mul
}

define i64 @m1(i64 %a) nounwind readnone {
entry:
; CHECK: dmult
; CHECK: mfhi
  %div = sdiv i64 %a, 3
  ret i64 %div
}

define i64 @d0(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: ddivu
; CHECK: mflo
  %div = udiv i64 %a0, %a1
  ret i64 %div
}

define i64 @d1(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: ddiv
; CHECK: mflo
  %div = sdiv i64 %a0, %a1
  ret i64 %div
}

define i64 @d2(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: ddivu
; CHECK: mfhi
  %rem = urem i64 %a0, %a1
  ret i64 %rem
}

define i64 @d3(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: ddiv
; CHECK: mfhi
  %rem = srem i64 %a0, %a1
  ret i64 %rem
}
