; RUN: llc < %s | FileCheck %s

; CHECK:	.globl x
; CHECK: x:
; CHECK: .quad	10

@x = global b64 bitcast (i64 10 to b64)

; CHECK:	.globl b
; CHECK: b:
; CHECK: .byte	1

@b = global b1 1

; CHECK:	.globl f
; CHECK: f:
; CHECK: .byte	31

@f = global b5 31

; CHECK:	.globl r
; CHECK: r:
; CHECK: .long	42

@r = global b32 42

; CHECK:	.globl w
; CHECK: w:
; CHECK: .quad	-1
; CHECK: .quad	-1

@w = global b128 -1

; CHECK:	.globl uw
; CHECK: uw:
; CHECK: .quad	-1
; CHECK: .quad	-1
; CHECK: .quad	-1
; CHECK: .quad	-1

@uw = global b256 -1

; CHECK:	.globl v
; CHECK: v:
; CHECK: .byte	1
; CHECK: .byte	2
; CHECK: .byte	3
; CHECK: .byte	4

@v = global <4 x b8> <b8 1, b8 2, b8 3, b8 4>

; CHECK:	.globl uv
; CHECK: uv:
; CHECK: .quad	-1
; CHECK: .quad	-1
; CHECK: .quad	-1
; CHECK: .quad	-1
; CHECK: .quad	-1
; CHECK: .quad	-1
; CHECK: .quad	-1
; CHECK: .quad	-1

@uv = global <4 x b128> <b128 -1, b128 -1, b128 -1, b128 -1>
