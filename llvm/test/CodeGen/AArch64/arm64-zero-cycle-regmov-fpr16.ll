; RUN: llc < %s -march=arm64 | FileCheck %s -check-prefixes=NOTCPU --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=apple-m1 | FileCheck %s -check-prefixes=CPU --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=apple-m1 -mattr=-zcm | FileCheck %s -check-prefixes=NOTATTR --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mattr=+zcm | FileCheck %s -check-prefixes=ATTR --match-full-lines

define void @t(half %a, half %b, half %c, half %d) {
entry:
; CHECK-LABEL: t:
; NOTCPU: fmov s0, s2
; NOTCPU: fmov s1, s3
; NOTCPU: fmov [[REG2:s[0-9]+]], s3
; NOTCPU: fmov [[REG1:s[0-9]+]], s2
; NOTCPU-NEXT: bl {{_?foo}}
; NOTCPU: fmov s0, [[REG1]]
; NOTCPU: fmov s1, [[REG2]]

; CPU: fmov [[REG2:d[0-9]+]], d3
; CPU: fmov [[REG1:d[0-9]+]], d2
; CPU: fmov d0, d2
; CPU: fmov d1, d3
; CPU-NEXT: bl {{_?foo}}
; CPU: fmov d0, [[REG1]]
; CPU: fmov d1, [[REG2]]

; NOTATTR: fmov [[REG2:s[0-9]+]], s3
; NOTATTR: fmov [[REG1:s[0-9]+]], s2
; NOTATTR: fmov s0, s2
; NOTATTR: fmov s1, s3
; NOTATTR-NEXT: bl {{_?foo}}
; NOTATTR: fmov s0, [[REG1]]
; NOTATTR: fmov s1, [[REG2]]

; ATTR: fmov d0, d2
; ATTR: fmov d1, d3
; ATTR: fmov [[REG2:d[0-9]+]], d3
; ATTR: fmov [[REG1:d[0-9]+]], d2
; ATTR-NEXT: bl {{_?foo}}
; ATTR: fmov d0, [[REG1]]
; ATTR: fmov d1, [[REG2]]
  %call = call half @foo(half %c, half %d)
  %call1 = call half @foo(half %c, half %d)
  unreachable
}

declare half @foo(half, half)
