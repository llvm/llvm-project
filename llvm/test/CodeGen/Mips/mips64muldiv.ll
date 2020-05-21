; RUN: llc -march=mips64el -mcpu=mips4 < %s | FileCheck %s -check-prefixes=ALL,ACC
; RUN: llc -march=mips64el -mcpu=mips64 < %s | FileCheck %s -check-prefixes=ALL,ACC
; RUN: llc -march=mips64el -mcpu=mips64r2 < %s | FileCheck %s -check-prefixes=ALL,ACC
; RUN: llc -march=mips64el -mcpu=mips64r6 < %s | FileCheck %s -check-prefixes=ALL,GPR

; COM: FileCheck prefixes:
; COM:  ALL - All targets
; COM:  ACC - Targets with accumulator based mul/div (i.e. pre-MIPS32r6)
; COM:  GPR - Targets with register based mul/div (i.e. MIPS32r6)

define i64 @m0(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: m0:
; ACC:           dmult ${{[45]}}, ${{[45]}}
; ACC:           mflo $2
; GPR:           dmul $2, ${{[45]}}, ${{[45]}}
  %mul = mul i64 %a1, %a0
  ret i64 %mul
}

define i64 @m1(i64 %a) nounwind readnone {
entry:
; ALL-LABEL: m1:
; ALL:           lui $[[T0:[0-9]+]], 21845
; ALL:           addiu $[[T0]], $[[T0]], 21845
; ALL:           dsll $[[T0]], $[[T0]], 16
; ALL:           addiu $[[T0]], $[[T0]], 21845
; ALL:           dsll $[[T0]], $[[T0]], 16
; ALL:           addiu $[[T0]], $[[T0]], 21846

; ACC:           dmult $4, $[[T0]]
; ACC:           mfhi $[[T1:[0-9]+]]
; GPR:           dmuh $[[T1:[0-9]+]], $4, $[[T0]]

; ALL:           dsrl $2, $[[T1]], 63
; ALL:           daddu $2, $[[T1]], $2
  %div = sdiv i64 %a, 3
  ret i64 %div
}

define i64 @d0(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: d0:
; ACC:           ddivu $zero, $4, $5
; ACC:           mflo $2
; GPR:           ddivu $2, $4, $5
  %div = udiv i64 %a0, %a1
  ret i64 %div
}

define i64 @d1(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: d1:
; ACC:           ddiv $zero, $4, $5
; ACC:           mflo $2
; GPR:           ddiv $2, $4, $5
  %div = sdiv i64 %a0, %a1
  ret i64 %div
}

define i64 @d2(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: d2:
; ACC:           ddivu $zero, $4, $5
; ACC:           mfhi $2
; GPR:           dmodu $2, $4, $5
  %rem = urem i64 %a0, %a1
  ret i64 %rem
}

define i64 @d3(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: d3:
; ACC:           ddiv $zero, $4, $5
; ACC:           mfhi $2
; GPR:           dmod $2, $4, $5
  %rem = srem i64 %a0, %a1
  ret i64 %rem
}
