; RUN: llc -mtriple=bpfel -mcpu=v5 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=V5,COMMON --implicit-check-not="/="
; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=NO-HMUL,COMMON --implicit-check-not="uh*=" --implicit-check-not="sh*="
; RUN: llc -mtriple=bpfel -mcpu=v5 -disable-hmul -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=NO-HMUL,COMMON --implicit-check-not="uh*=" --implicit-check-not="sh*="

; Constant division can introduce high-half multiplication even when the IR
; contains no multiply. Only v5 with HMUL enabled may use the new instructions.

define i64 @udiv_by_10(i64 %a) {
; V5-LABEL: udiv_by_10:
; V5:       r[[REG1:[0-9]+]] = -3689348814741910323 ll
; V5:       r0 uh*= r[[REG1]]
; V5-NEXT:  r0 >>= 3
; V5-NEXT:  exit
; NO-HMUL-LABEL: udiv_by_10:
; NO-HMUL:       r0 = r1
; NO-HMUL-NEXT:  r0 /= 10
; NO-HMUL-NEXT:  exit
  %result = udiv i64 %a, 10
  ret i64 %result
}

; Signed division needs a correction so negative quotients round toward zero.
define i64 @sdiv_by_10(i64 %a) {
; V5-LABEL: sdiv_by_10:
; V5:       r[[REG1:[0-9]+]] = 7378697629483820647 ll
; V5:       r[[REG2:[0-9]+]] sh*= r[[REG1]]
; V5:       r0 = r[[REG2]]
; V5-DAG:   r0 >>= 63
; V5-DAG:   r[[REG2]] s>>= 2
; V5-NEXT:  r0 += r[[REG2]]
; V5-NEXT:  exit
; NO-HMUL-LABEL: sdiv_by_10:
; NO-HMUL:       r0 = r1
; NO-HMUL-NEXT:  r0 s/= 10
; NO-HMUL-NEXT:  exit
  %result = sdiv i64 %a, 10
  ret i64 %result
}

define i64 @sdiv_by_neg10(i64 %a) {
; V5-LABEL: sdiv_by_neg10:
; V5:       r[[REG1:[0-9]+]] = -7378697629483820647 ll
; V5:       r[[REG2:[0-9]+]] sh*= r[[REG1]]
; V5:       r0 = r[[REG2]]
; V5-DAG:   r0 >>= 63
; V5-DAG:   r[[REG2]] s>>= 2
; V5-NEXT:  r0 += r[[REG2]]
; V5-NEXT:  exit
; NO-HMUL-LABEL: sdiv_by_neg10:
; NO-HMUL:       r0 = r1
; NO-HMUL-NEXT:  r0 s/= -10
; NO-HMUL-NEXT:  exit
  %result = sdiv i64 %a, -10
  ret i64 %result
}

; Division by a power of two remains a shift in all configurations.
define i64 @udiv_by_8(i64 %a) {
; COMMON-LABEL: udiv_by_8:
; COMMON:       r0 = r1
; COMMON-NEXT:  r0 >>= 3
; COMMON-NEXT:  exit
  %result = udiv i64 %a, 8
  ret i64 %result
}
