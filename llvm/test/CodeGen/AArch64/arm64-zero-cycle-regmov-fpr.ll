; RUN: llc < %s -mtriple=arm64-linux-gnu | FileCheck %s -check-prefixes=NOTCPU-LINUX --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=generic | FileCheck %s -check-prefixes=NOTCPU-APPLE --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=apple-m1 | FileCheck %s -check-prefixes=CPU --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=apple-m1 -mattr=-zcm-fpr128 | FileCheck %s -check-prefixes=NOTATTR --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mattr=+zcm-fpr128 | FileCheck %s -check-prefixes=ATTR --match-full-lines

define void @zero_cycle_regmov_FPR64(double %a, double %b, double %c, double %d) {
entry:
; CHECK-LABEL: t:
; NOTCPU-LINUX: fmov d0, d2
; NOTCPU-LINUX: fmov d1, d3
; NOTCPU-LINUX: fmov [[REG2:d[0-9]+]], d3
; NOTCPU-LINUX: fmov [[REG1:d[0-9]+]], d2
; NOTCPU-LINUX-NEXT: bl {{_?foo_double}}
; NOTCPU-LINUX: fmov d0, [[REG1]]
; NOTCPU-LINUX: fmov d1, [[REG2]]

; NOTCPU-APPLE: fmov d0, d2
; NOTCPU-APPLE: fmov d1, d3
; NOTCPU-APPLE: fmov [[REG2:d[0-9]+]], d3
; NOTCPU-APPLE: fmov [[REG1:d[0-9]+]], d2
; NOTCPU-APPLE-NEXT: bl {{_?foo_double}}
; NOTCPU-APPLE: fmov d0, [[REG1]]
; NOTCPU-APPLE: fmov d1, [[REG2]]

; CPU: mov.16b [[REG2:v[0-9]+]], v3
; CPU: mov.16b [[REG1:v[0-9]+]], v2
; CPU: mov.16b v0, v2
; CPU: mov.16b v1, v3
; CPU-NEXT: bl {{_?foo_double}}
; CPU: mov.16b v0, [[REG1]]
; CPU: mov.16b v1, [[REG2]]

; NOTATTR: fmov [[REG2:d[0-9]+]], d3
; NOTATTR: fmov [[REG1:d[0-9]+]], d2
; NOTATTR: fmov d0, d2
; NOTATTR: fmov d1, d3
; NOTATTR-NEXT: bl {{_?foo_double}}
; NOTATTR: fmov d0, [[REG1]]
; NOTATTR: fmov d1, [[REG2]]

; ATTR: mov.16b v0, v2
; ATTR: mov.16b v1, v3
; ATTR: mov.16b [[REG2:v[0-9]+]], v3
; ATTR: mov.16b [[REG1:v[0-9]+]], v2
; ATTR-NEXT: bl {{_?foo_double}}
; ATTR: mov.16b v0, [[REG1]]
; ATTR: mov.16b v1, [[REG2]]
  %call = call double @foo_double(double %c, double %d)
  %call1 = call double @foo_double(double %c, double %d)
  unreachable
}

declare float @foo_double(double, double)

define void @zero_cycle_regmov_FPR32(float %a, float %b, float %c, float %d) {
entry:
; CHECK-LABEL: t:
; NOTCPU-LINUX: fmov s0, s2
; NOTCPU-LINUX: fmov s1, s3
; NOTCPU-LINUX: fmov [[REG2:s[0-9]+]], s3
; NOTCPU-LINUX: fmov [[REG1:s[0-9]+]], s2
; NOTCPU-LINUX-NEXT: bl {{_?foo_float}}
; NOTCPU-LINUX: fmov s0, [[REG1]]
; NOTCPU-LINUX: fmov s1, [[REG2]]

; NOTCPU-APPLE: fmov s0, s2
; NOTCPU-APPLE: fmov s1, s3
; NOTCPU-APPLE: fmov [[REG2:s[0-9]+]], s3
; NOTCPU-APPLE: fmov [[REG1:s[0-9]+]], s2
; NOTCPU-APPLE-NEXT: bl {{_?foo_float}}
; NOTCPU-APPLE: fmov s0, [[REG1]]
; NOTCPU-APPLE: fmov s1, [[REG2]]

; CPU: mov.16b [[REG2:v[0-9]+]], v3
; CPU: mov.16b [[REG1:v[0-9]+]], v2
; CPU: mov.16b v0, v2
; CPU: mov.16b v1, v3
; CPU-NEXT: bl {{_?foo_float}}
; CPU: mov.16b v0, [[REG1]]
; CPU: mov.16b v1, [[REG2]]

; NOTATTR: fmov [[REG2:s[0-9]+]], s3
; NOTATTR: fmov [[REG1:s[0-9]+]], s2
; NOTATTR: fmov s0, s2
; NOTATTR: fmov s1, s3
; NOTATTR-NEXT: bl {{_?foo_float}}
; NOTATTR: fmov s0, [[REG1]]
; NOTATTR: fmov s1, [[REG2]]

; ATTR: mov.16b v0, v2
; ATTR: mov.16b v1, v3
; ATTR: mov.16b [[REG2:v[0-9]+]], v3
; ATTR: mov.16b [[REG1:v[0-9]+]], v2
; ATTR-NEXT: bl {{_?foo_float}}
; ATTR: mov.16b v0, [[REG1]]
; ATTR: mov.16b v1, [[REG2]]
  %call = call float @foo_float(float %c, float %d)
  %call1 = call float @foo_float(float %c, float %d)
  unreachable
}

declare float @foo_float(float, float)

define void @zero_cycle_regmov_FPR16(half %a, half %b, half %c, half %d) {
entry:
; CHECK-LABEL: t:
; NOTCPU-LINUX: fmov s0, s2
; NOTCPU-LINUX: fmov s1, s3
; NOTCPU-LINUX: fmov [[REG2:s[0-9]+]], s3
; NOTCPU-LINUX: fmov [[REG1:s[0-9]+]], s2
; NOTCPU-LINUX-NEXT: bl {{_?foo_half}}
; NOTCPU-LINUX: fmov s0, [[REG1]]
; NOTCPU-LINUX: fmov s1, [[REG2]]

; NOTCPU-APPLE: fmov s0, s2
; NOTCPU-APPLE: fmov s1, s3
; NOTCPU-APPLE: fmov [[REG2:s[0-9]+]], s3
; NOTCPU-APPLE: fmov [[REG1:s[0-9]+]], s2
; NOTCPU-APPLE-NEXT: bl {{_?foo_half}}
; NOTCPU-APPLE: fmov s0, [[REG1]]
; NOTCPU-APPLE: fmov s1, [[REG2]]

; CPU: mov.16b [[REG2:v[0-9]+]], v3
; CPU: mov.16b [[REG1:v[0-9]+]], v2
; CPU: mov.16b v0, v2
; CPU: mov.16b v1, v3
; CPU-NEXT: bl {{_?foo_half}}
; CPU: mov.16b v0, [[REG1]]
; CPU: mov.16b v1, [[REG2]]

; NOTATTR: fmov [[REG2:s[0-9]+]], s3
; NOTATTR: fmov [[REG1:s[0-9]+]], s2
; NOTATTR: fmov s0, s2
; NOTATTR: fmov s1, s3
; NOTATTR-NEXT: bl {{_?foo_half}}
; NOTATTR: fmov s0, [[REG1]]
; NOTATTR: fmov s1, [[REG2]]

; ATTR: mov.16b v0, v2
; ATTR: mov.16b v1, v3
; ATTR: mov.16b [[REG2:v[0-9]+]], v3
; ATTR: mov.16b [[REG1:v[0-9]+]], v2
; ATTR-NEXT: bl {{_?foo_half}}
; ATTR: mov.16b v0, [[REG1]]
; ATTR: mov.16b v1, [[REG2]]
  %call = call half @foo_half(half %c, half %d)
  %call1 = call half @foo_half(half %c, half %d)
  unreachable
}

declare half @foo_half(half, half)
