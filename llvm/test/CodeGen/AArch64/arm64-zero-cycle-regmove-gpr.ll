; RUN: llc < %s -mtriple=arm64-linux-gnu | FileCheck %s -check-prefixes=NOTCPU-LINUX --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=generic | FileCheck %s -check-prefixes=NOTCPU-APPLE --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=apple-m1 | FileCheck %s -check-prefixes=CPU --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=apple-m1 -mattr=-zcm-gpr64 | FileCheck %s -check-prefixes=NOTATTR --match-full-lines
; RUN: llc < %s -mtriple=arm64-apple-macosx -mattr=+zcm-gpr64 | FileCheck %s -check-prefixes=ATTR --match-full-lines

define void @zero_cycle_regmove_GPR32(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
; CHECK-LABEL: t:
; NOTCPU-LINUX: mov w0, w2
; NOTCPU-LINUX: mov w1, w3
; NOTCPU-LINUX: mov [[REG2:w[0-9]+]], w3
; NOTCPU-LINUX: mov [[REG1:w[0-9]+]], w2
; NOTCPU-LINUX-NEXT: bl {{_?foo_i32}}
; NOTCPU-LINUX: mov w0, [[REG1]]
; NOTCPU-LINUX: mov w1, [[REG2]]

; NOTCPU-APPLE: mov w0, w2
; NOTCPU-APPLE: mov w1, w3
; NOTCPU-APPLE: mov [[REG2:w[0-9]+]], w3
; NOTCPU-APPLE: mov [[REG1:w[0-9]+]], w2
; NOTCPU-APPLE-NEXT: bl {{_?foo_i32}}
; NOTCPU-APPLE: mov w0, [[REG1]]
; NOTCPU-APPLE: mov w1, [[REG2]]

; CPU: mov [[REG2:x[0-9]+]], x3
; CPU: mov [[REG1:x[0-9]+]], x2
; CPU: mov x0, x2
; CPU: mov x1, x3
; CPU-NEXT: bl {{_?foo_i32}}
; CPU: mov x0, [[REG1]]
; CPU: mov x1, [[REG2]]

; NOTATTR: mov [[REG2:w[0-9]+]], w3
; NOTATTR: mov [[REG1:w[0-9]+]], w2
; NOTATTR: mov w0, w2
; NOTATTR: mov w1, w3
; NOTATTR-NEXT: bl {{_?foo_i32}}
; NOTATTR: mov w0, [[REG1]]
; NOTATTR: mov w1, [[REG2]]

; ATTR: mov x0, x2
; ATTR: mov x1, x3
; ATTR: mov [[REG2:x[0-9]+]], x3
; ATTR: mov [[REG1:x[0-9]+]], x2
; ATTR-NEXT: bl {{_?foo_i32}}
; ATTR: mov x0, [[REG1]]
; ATTR: mov x1, [[REG2]]
  %call = call i32 @foo_i32(i32 %c, i32 %d)
  %call1 = call i32 @foo_i32(i32 %c, i32 %d)
  unreachable
}

declare i32 @foo_i32(i32, i32)
