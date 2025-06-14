; RUN: llc < %s -march=arm64 | FileCheck %s -check-prefixes=NOTCPU
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=apple-m1 | FileCheck %s -check-prefixes=CPU
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=apple-m1 -mattr=-zcm | FileCheck %s -check-prefixes=NOTATTR
; RUN: llc < %s -mtriple=arm64-apple-macosx -mattr=+zcm | FileCheck %s -check-prefixes=ATTR

define half @t(half %a, half %b, half %c, half %d) {
entry:
; CHECK-LABEL: t:
; NOTCPU: mov s0, s2
; NOTCPU: mov s1, s3
; NOTCPU: mov [[REG2:s[0-9]+]], s3
; NOTCPU: mov [[REG1:s[0-9]+]], s2
; NOTCPU: bl {{_?foo}}
; NOTCPU: mov s0, [[REG1]]
; NOTCPU: mov s1, [[REG2]]

; CPU: mov [[REG2:d[0-9]+]], d3
; CPU: mov [[REG1:d[0-9]+]], d2
; CPU: mov d0, d2
; CPU: mov d1, d3
; CPU: bl {{_?foo}}
; CPU: mov d0, [[REG1]]
; CPU: mov d1, [[REG2]]

; NOTATTR: mov [[REG2:s[0-9]+]], s3
; NOTATTR: mov [[REG1:s[0-9]+]], s2
; NOTATTR: mov s0, s2
; NOTATTR: mov s1, s3
; NOTATTR: bl {{_?foo}}
; NOTATTR: mov s0, [[REG1]]
; NOTATTR: mov s1, [[REG2]]

; ATTR: mov d0, d2
; ATTR: mov d1, d3
; ATTR: mov [[REG2:d[0-9]+]], d3
; ATTR: mov [[REG1:d[0-9]+]], d2
; ATTR: bl {{_?foo}}
; ATTR: mov d0, [[REG1]]
; ATTR: mov d1, [[REG2]]
  %call = call half @foo(half %c, half %d)
  %call1 = call half @foo(half %c, half %d)
  unreachable
}

declare half @foo(half, half)
