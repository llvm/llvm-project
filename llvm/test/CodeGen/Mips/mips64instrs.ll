; RUN: llc -march=mips64el -mcpu=mips64 < %s | FileCheck %s

define i64 @f0(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: daddu
  %add = add nsw i64 %a1, %a0
  ret i64 %add
}

define i64 @f1(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: dsubu
  %sub = sub nsw i64 %a0, %a1
  ret i64 %sub
}

define i64 @f4(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: and
  %and = and i64 %a1, %a0
  ret i64 %and
}

define i64 @f5(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: or
  %or = or i64 %a1, %a0
  ret i64 %or
}

define i64 @f6(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: xor
  %xor = xor i64 %a1, %a0
  ret i64 %xor
}

define i64 @f7(i64 %a0) nounwind readnone {
entry:
; CHECK: daddiu ${{[0-9]+}}, ${{[0-9]+}}, 20
  %add = add nsw i64 %a0, 20
  ret i64 %add
}

define i64 @f8(i64 %a0) nounwind readnone {
entry:
; CHECK: daddiu ${{[0-9]+}}, ${{[0-9]+}}, -20
  %sub = add nsw i64 %a0, -20
  ret i64 %sub
}

define i64 @f9(i64 %a0) nounwind readnone {
entry:
; CHECK: andi ${{[0-9]+}}, ${{[0-9]+}}, 20
  %and = and i64 %a0, 20
  ret i64 %and
}

define i64 @f10(i64 %a0) nounwind readnone {
entry:
; CHECK: ori ${{[0-9]+}}, ${{[0-9]+}}, 20
  %or = or i64 %a0, 20
  ret i64 %or
}

define i64 @f11(i64 %a0) nounwind readnone {
entry:
; CHECK: xori ${{[0-9]+}}, ${{[0-9]+}}, 20
  %xor = xor i64 %a0, 20
  ret i64 %xor
}

define i64 @f12(i64 %a, i64 %b) nounwind readnone {
entry:
; CHECK: mult
  %mul = mul nsw i64 %b, %a
  ret i64 %mul
}

define i64 @f13(i64 %a, i64 %b) nounwind readnone {
entry:
; CHECK: mult
  %mul = mul i64 %b, %a
  ret i64 %mul
}

define i64 @f14(i64 %a, i64 %b) nounwind readnone {
entry:
; CHECK: ddiv $zero
; CHECK: mflo
  %div = sdiv i64 %a, %b
  ret i64 %div
}

define i64 @f15(i64 %a, i64 %b) nounwind readnone {
entry:
; CHECK: ddivu $zero
; CHECK: mflo
  %div = udiv i64 %a, %b
  ret i64 %div
}

define i64 @f16(i64 %a, i64 %b) nounwind readnone {
entry:
; CHECK: ddiv $zero
; CHECK: mfhi
  %rem = srem i64 %a, %b
  ret i64 %rem
}

define i64 @f17(i64 %a, i64 %b) nounwind readnone {
entry:
; CHECK: ddivu $zero
; CHECK: mfhi
  %rem = urem i64 %a, %b
  ret i64 %rem
}

declare i64 @llvm.ctlz.i64(i64, i1) nounwind readnone

define i64 @f18(i64 %X) nounwind readnone {
entry:
; CHECK: dclz $2, $4
  %tmp1 = tail call i64 @llvm.ctlz.i64(i64 %X, i1 true)
  ret i64 %tmp1
}

define i64 @f19(i64 %X) nounwind readnone {
entry:
; CHECK: dclo $2, $4
  %neg = xor i64 %X, -1
  %tmp1 = tail call i64 @llvm.ctlz.i64(i64 %neg, i1 true)
  ret i64 %tmp1
}

define i64 @f20(i64 %a, i64 %b) nounwind readnone {
entry:
; CHECK: nor
  %or = or i64 %b, %a
  %neg = xor i64 %or, -1
  ret i64 %neg
}

