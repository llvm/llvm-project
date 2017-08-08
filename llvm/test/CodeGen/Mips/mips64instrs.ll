; RUN: llc -march=mips64el -mcpu=mips4 -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,MIPS4,ACCMULDIV %s
; RUN: llc -march=mips64el -mcpu=mips64 -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,HAS-DCLO,ACCMULDIV %s
; RUN: llc -march=mips64el -mcpu=mips64r2 -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,HAS-DCLO,ACCMULDIV %s
; RUN: llc -march=mips64el -mcpu=mips64r6 -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,HAS-DCLO,GPRMULDIV %s

@gll0 = common global i64 0, align 8
@gll1 = common global i64 0, align 8

define i64 @f0(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: f0:
; ALL:           daddu $2, ${{[45]}}, ${{[45]}}
  %add = add nsw i64 %a1, %a0
  ret i64 %add
}

define i64 @f1(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: f1:
; ALL:           dsubu $2, $4, $5
  %sub = sub nsw i64 %a0, %a1
  ret i64 %sub
}

define i64 @f4(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: f4:
; ALL:           and $2, ${{[45]}}, ${{[45]}}
  %and = and i64 %a1, %a0
  ret i64 %and
}

define i64 @f5(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: f5:
; ALL:           or $2, ${{[45]}}, ${{[45]}}
  %or = or i64 %a1, %a0
  ret i64 %or
}

define i64 @f6(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: f6:
; ALL:           xor $2, ${{[45]}}, ${{[45]}}
  %xor = xor i64 %a1, %a0
  ret i64 %xor
}

define i64 @f7(i64 %a0) nounwind readnone {
entry:
; ALL-LABEL: f7:
; ALL:           daddiu $2, $4, 20
  %add = add nsw i64 %a0, 20
  ret i64 %add
}

define i64 @f8(i64 %a0) nounwind readnone {
entry:
; ALL-LABEL: f8:
; ALL:           daddiu $2, $4, -20
  %sub = add nsw i64 %a0, -20
  ret i64 %sub
}

define i64 @f9(i64 %a0) nounwind readnone {
entry:
; ALL-LABEL: f9:
; ALL:           andi $2, $4, 20
  %and = and i64 %a0, 20
  ret i64 %and
}

define i64 @f10(i64 %a0) nounwind readnone {
entry:
; ALL-LABEL: f10:
; ALL:           ori $2, $4, 20
  %or = or i64 %a0, 20
  ret i64 %or
}

define i64 @f11(i64 %a0) nounwind readnone {
entry:
; ALL-LABEL: f11:
; ALL:           xori $2, $4, 20
  %xor = xor i64 %a0, 20
  ret i64 %xor
}

define i64 @f12(i64 %a, i64 %b) nounwind readnone {
entry:
; ALL-LABEL: f12:

; ACCMULDIV:     mult ${{[45]}}, ${{[45]}}
; GPRMULDIV:     dmul $2, ${{[45]}}, ${{[45]}}

  %mul = mul nsw i64 %b, %a
  ret i64 %mul
}

define i64 @f13(i64 %a, i64 %b) nounwind readnone {
entry:
; ALL-LABEL: f13:

; ACCMULDIV:     mult ${{[45]}}, ${{[45]}}
; GPRMULDIV:     dmul $2, ${{[45]}}, ${{[45]}}

  %mul = mul i64 %b, %a
  ret i64 %mul
}

define i64 @f14(i64 %a, i64 %b) nounwind readnone {
entry:
; ALL-LABEL: f14:
; ALL-DAG:       ld $[[T0:[0-9]+]], %lo(gll0)(${{[0-9]+}})
; ALL-DAG:       ld $[[T1:[0-9]+]], %lo(gll1)(${{[0-9]+}})

; ACCMULDIV:     ddiv $zero, $[[T0]], $[[T1]]
; ACCMULDIV:     teq $[[T1]], $zero, 7
; ACCMULDIV:     mflo $2

; GPRMULDIV:     ddiv $2, $[[T0]], $[[T1]]
; GPRMULDIV:     teq $[[T1]], $zero, 7

  %0 = load i64, i64* @gll0, align 8
  %1 = load i64, i64* @gll1, align 8
  %div = sdiv i64 %0, %1
  ret i64 %div
}

define i64 @f15() nounwind readnone {
entry:
; ALL-LABEL: f15:
; ALL-DAG:       ld $[[T0:[0-9]+]], %lo(gll0)(${{[0-9]+}})
; ALL-DAG:       ld $[[T1:[0-9]+]], %lo(gll1)(${{[0-9]+}})

; ACCMULDIV:     ddivu $zero, $[[T0]], $[[T1]]
; ACCMULDIV:     teq $[[T1]], $zero, 7
; ACCMULDIV:     mflo $2

; GPRMULDIV:     ddivu $2, $[[T0]], $[[T1]]
; GPRMULDIV:     teq $[[T1]], $zero, 7

  %0 = load i64, i64* @gll0, align 8
  %1 = load i64, i64* @gll1, align 8
  %div = udiv i64 %0, %1
  ret i64 %div
}

define i64 @f16(i64 %a, i64 %b) nounwind readnone {
entry:
; ALL-LABEL: f16:

; ACCMULDIV:     ddiv $zero, $4, $5
; ACCMULDIV:     teq $5, $zero, 7
; ACCMULDIV:     mfhi $2

; GPRMULDIV:     dmod $2, $4, $5
; GPRMULDIV:     teq $5, $zero, 7

  %rem = srem i64 %a, %b
  ret i64 %rem
}

define i64 @f17(i64 %a, i64 %b) nounwind readnone {
entry:
; ALL-LABEL: f17:

; ACCMULDIV:     ddivu $zero, $4, $5
; ACCMULDIV:     teq $5, $zero, 7
; ACCMULDIV:     mfhi $2

; GPRMULDIV:     dmodu $2, $4, $5
; GPRMULDIV:     teq $5, $zero, 7

  %rem = urem i64 %a, %b
  ret i64 %rem
}

declare i64 @llvm.ctlz.i64(i64, i1) nounwind readnone

define i64 @f18(i64 %X) nounwind readnone {
entry:
; ALL-LABEL: f18:

; The MIPS4 version is too long to reasonably test. At least check we don't get dclz
; MIPS4-NOT:     dclz

; HAS-DCLO:      dclz $2, $4

  %tmp1 = tail call i64 @llvm.ctlz.i64(i64 %X, i1 true)
  ret i64 %tmp1
}

define i64 @f19(i64 %X) nounwind readnone {
entry:
; ALL-LABEL: f19:

; The MIPS4 version is too long to reasonably test. At least check we don't get dclo
; MIPS4-NOT:     dclo

; HAS-DCLO:      dclo $2, $4

  %neg = xor i64 %X, -1
  %tmp1 = tail call i64 @llvm.ctlz.i64(i64 %neg, i1 true)
  ret i64 %tmp1
}

define i64 @f20(i64 %a, i64 %b) nounwind readnone {
entry:
; ALL-LABEL: f20:
; ALL:           nor $2, ${{[45]}}, ${{[45]}}
  %or = or i64 %b, %a
  %neg = xor i64 %or, -1
  ret i64 %neg
}
