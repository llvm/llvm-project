; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s

foo:

; Flag set operations

; SEC
bset 0
sec

; SEZ
bset 1
sez

; SEN
bset 2
sen

; SEV
bset 3
sev

; SES
bset 4
ses

; SEH
bset 5
seh

; SET
bset 6
set

; SEI
bset 7
sei

; Flag clear operations

; CLC
bclr 0
clc

; CLZ
bclr 1
clz

; CLN
bclr 2
cln

; CLV
bclr 3
clv

; CLS
bclr 4
cls

; CLH
bclr 5
clh

; CLT
bclr 6
clt

; CLI
bclr 7
cli

; CHECK: sec               ; encoding: [0x08,0x94]
; CHECK: sec               ; encoding: [0x08,0x94]
; CHECK: sez               ; encoding: [0x18,0x94]
; CHECK: sez               ; encoding: [0x18,0x94]
; CHECK: sen               ; encoding: [0x28,0x94]
; CHECK: sen               ; encoding: [0x28,0x94]
; CHECK: sev               ; encoding: [0x38,0x94]
; CHECK: sev               ; encoding: [0x38,0x94]
; CHECK: ses               ; encoding: [0x48,0x94]
; CHECK: ses               ; encoding: [0x48,0x94]
; CHECK: seh               ; encoding: [0x58,0x94]
; CHECK: seh               ; encoding: [0x58,0x94]
; CHECK: set               ; encoding: [0x68,0x94]
; CHECK: set               ; encoding: [0x68,0x94]
; CHECK: sei               ; encoding: [0x78,0x94]
; CHECK: sei               ; encoding: [0x78,0x94]

; CHECK: clc               ; encoding: [0x88,0x94]
; CHECK: clc               ; encoding: [0x88,0x94]
; CHECK: clz               ; encoding: [0x98,0x94]
; CHECK: clz               ; encoding: [0x98,0x94]
; CHECK: cln               ; encoding: [0xa8,0x94]
; CHECK: cln               ; encoding: [0xa8,0x94]
; CHECK: clv               ; encoding: [0xb8,0x94]
; CHECK: clv               ; encoding: [0xb8,0x94]
; CHECK: cls               ; encoding: [0xc8,0x94]
; CHECK: cls               ; encoding: [0xc8,0x94]
; CHECK: clh               ; encoding: [0xd8,0x94]
; CHECK: clh               ; encoding: [0xd8,0x94]
; CHECK: clt               ; encoding: [0xe8,0x94]
; CHECK: clt               ; encoding: [0xe8,0x94]
; CHECK: cli               ; encoding: [0xf8,0x94]
; CHECK: cli               ; encoding: [0xf8,0x94]
