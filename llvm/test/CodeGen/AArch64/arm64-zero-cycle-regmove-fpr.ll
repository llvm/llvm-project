; RUN: llc < %s -mtriple=arm64-linux-gnu | FileCheck %s -check-prefixes=NOZCM-FPR128-CPU --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=generic | FileCheck %s -check-prefixes=NOZCM-FPR128-CPU --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=apple-m1 | FileCheck %s -check-prefixes=ZCM-FPR128-CPU --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=apple-m1 -mattr=-zcm-fpr128 | FileCheck %s -check-prefixes=NOZCM-FPR128-ATTR --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mattr=+zcm-fpr128 | FileCheck %s -check-prefixes=ZCM-FPR128-ATTR --match-full-lines

define void @zero_cycle_regmove_FPR64(double %a, double %b, double %c, double %d) {
entry:
; CHECK-LABEL: t:
; NOZCM-FPR128-CPU: fmov d0, d2
; NOZCM-FPR128-CPU: fmov d1, d3
; NOZCM-FPR128-CPU: fmov [[REG2:d[0-9]+]], d3
; NOZCM-FPR128-CPU: fmov [[REG1:d[0-9]+]], d2
; NOZCM-FPR128-CPU-NEXT: bl {{_?foo_double}}
; NOZCM-FPR128-CPU: fmov d0, [[REG1]]
; NOZCM-FPR128-CPU: fmov d1, [[REG2]]

; ZCM-FPR128-CPU: mov.16b [[REG2:v[0-9]+]], v3
; ZCM-FPR128-CPU: mov.16b [[REG1:v[0-9]+]], v2
; ZCM-FPR128-CPU: mov.16b v0, v2
; ZCM-FPR128-CPU: mov.16b v1, v3
; ZCM-FPR128-CPU-NEXT: bl {{_?foo_double}}
; ZCM-FPR128-CPU: mov.16b v0, [[REG1]]
; ZCM-FPR128-CPU: mov.16b v1, [[REG2]]

; NOZCM-FPR128-ATTR: fmov [[REG2:d[0-9]+]], d3
; NOZCM-FPR128-ATTR: fmov [[REG1:d[0-9]+]], d2
; NOZCM-FPR128-ATTR: fmov d0, d2
; NOZCM-FPR128-ATTR: fmov d1, d3
; NOZCM-FPR128-ATTR-NEXT: bl {{_?foo_double}}
; NOZCM-FPR128-ATTR: fmov d0, [[REG1]]
; NOZCM-FPR128-ATTR: fmov d1, [[REG2]]

; ZCM-FPR128-ATTR: mov.16b v0, v2
; ZCM-FPR128-ATTR: mov.16b v1, v3
; ZCM-FPR128-ATTR: mov.16b [[REG2:v[0-9]+]], v3
; ZCM-FPR128-ATTR: mov.16b [[REG1:v[0-9]+]], v2
; ZCM-FPR128-ATTR-NEXT: bl {{_?foo_double}}
; ZCM-FPR128-ATTR: mov.16b v0, [[REG1]]
; ZCM-FPR128-ATTR: mov.16b v1, [[REG2]]
  %call = call double @foo_double(double %c, double %d)
  %call1 = call double @foo_double(double %c, double %d)
  unreachable
}

declare float @foo_double(double, double)

define void @zero_cycle_regmove_FPR32(float %a, float %b, float %c, float %d) {
entry:
; CHECK-LABEL: t:
; NOZCM-FPR128-CPU: fmov s0, s2
; NOZCM-FPR128-CPU: fmov s1, s3
; NOZCM-FPR128-CPU: fmov [[REG2:s[0-9]+]], s3
; NOZCM-FPR128-CPU: fmov [[REG1:s[0-9]+]], s2
; NOZCM-FPR128-CPU-NEXT: bl {{_?foo_float}}
; NOZCM-FPR128-CPU: fmov s0, [[REG1]]
; NOZCM-FPR128-CPU: fmov s1, [[REG2]]

; ZCM-FPR128-CPU: mov.16b [[REG2:v[0-9]+]], v3
; ZCM-FPR128-CPU: mov.16b [[REG1:v[0-9]+]], v2
; ZCM-FPR128-CPU: mov.16b v0, v2
; ZCM-FPR128-CPU: mov.16b v1, v3
; ZCM-FPR128-CPU-NEXT: bl {{_?foo_float}}
; ZCM-FPR128-CPU: mov.16b v0, [[REG1]]
; ZCM-FPR128-CPU: mov.16b v1, [[REG2]]

; NOZCM-FPR128-ATTR: fmov [[REG2:s[0-9]+]], s3
; NOZCM-FPR128-ATTR: fmov [[REG1:s[0-9]+]], s2
; NOZCM-FPR128-ATTR: fmov s0, s2
; NOZCM-FPR128-ATTR: fmov s1, s3
; NOZCM-FPR128-ATTR-NEXT: bl {{_?foo_float}}
; NOZCM-FPR128-ATTR: fmov s0, [[REG1]]
; NOZCM-FPR128-ATTR: fmov s1, [[REG2]]

; ZCM-FPR128-ATTR: mov.16b v0, v2
; ZCM-FPR128-ATTR: mov.16b v1, v3
; ZCM-FPR128-ATTR: mov.16b [[REG2:v[0-9]+]], v3
; ZCM-FPR128-ATTR: mov.16b [[REG1:v[0-9]+]], v2
; ZCM-FPR128-ATTR-NEXT: bl {{_?foo_float}}
; ZCM-FPR128-ATTR: mov.16b v0, [[REG1]]
; ZCM-FPR128-ATTR: mov.16b v1, [[REG2]]
  %call = call float @foo_float(float %c, float %d)
  %call1 = call float @foo_float(float %c, float %d)
  unreachable
}

declare float @foo_float(float, float)

define void @zero_cycle_regmove_FPR16(half %a, half %b, half %c, half %d) {
entry:
; CHECK-LABEL: t:
; NOZCM-FPR128-CPU: fmov s0, s2
; NOZCM-FPR128-CPU: fmov s1, s3
; NOZCM-FPR128-CPU: fmov [[REG2:s[0-9]+]], s3
; NOZCM-FPR128-CPU: fmov [[REG1:s[0-9]+]], s2
; NOZCM-FPR128-CPU-NEXT: bl {{_?foo_half}}
; NOZCM-FPR128-CPU: fmov s0, [[REG1]]
; NOZCM-FPR128-CPU: fmov s1, [[REG2]]

; ZCM-FPR128-CPU: mov.16b [[REG2:v[0-9]+]], v3
; ZCM-FPR128-CPU: mov.16b [[REG1:v[0-9]+]], v2
; ZCM-FPR128-CPU: mov.16b v0, v2
; ZCM-FPR128-CPU: mov.16b v1, v3
; ZCM-FPR128-CPU-NEXT: bl {{_?foo_half}}
; ZCM-FPR128-CPU: mov.16b v0, [[REG1]]
; ZCM-FPR128-CPU: mov.16b v1, [[REG2]]

; NOZCM-FPR128-ATTR: fmov [[REG2:s[0-9]+]], s3
; NOZCM-FPR128-ATTR: fmov [[REG1:s[0-9]+]], s2
; NOZCM-FPR128-ATTR: fmov s0, s2
; NOZCM-FPR128-ATTR: fmov s1, s3
; NOZCM-FPR128-ATTR-NEXT: bl {{_?foo_half}}
; NOZCM-FPR128-ATTR: fmov s0, [[REG1]]
; NOZCM-FPR128-ATTR: fmov s1, [[REG2]]

; ZCM-FPR128-ATTR: mov.16b v0, v2
; ZCM-FPR128-ATTR: mov.16b v1, v3
; ZCM-FPR128-ATTR: mov.16b [[REG2:v[0-9]+]], v3
; ZCM-FPR128-ATTR: mov.16b [[REG1:v[0-9]+]], v2
; ZCM-FPR128-ATTR-NEXT: bl {{_?foo_half}}
; ZCM-FPR128-ATTR: mov.16b v0, [[REG1]]
; ZCM-FPR128-ATTR: mov.16b v1, [[REG2]]
  %call = call half @foo_half(half %c, half %d)
  %call1 = call half @foo_half(half %c, half %d)
  unreachable
}

declare half @foo_half(half, half)
