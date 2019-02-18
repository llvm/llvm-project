; RUN: llvm-mc -triple msp430 -show-encoding < %s | FileCheck %s

foo:
  mov r8, r15
  mov disp+2(r8), r15
  mov disp+2, r15
  mov &disp+2, r15
  mov @r8, r15
  mov @r8+, r15
  mov #disp+2, r15

; CHECK: mov r8, r15           ; encoding: [0x0f,0x48]
; CHECK: mov disp+2(r8), r15   ; encoding: [0x1f,0x48,A,A]
; CHECK: mov disp+2, r15       ; encoding: [0x1f,0x40,A,A]
; CHECK: mov &disp+2, r15      ; encoding: [0x1f,0x42,A,A]
; CHECK: mov @r8, r15          ; encoding: [0x2f,0x48]
; CHECK: mov @r8+, r15         ; encoding: [0x3f,0x48]
; CHECK: mov #disp+2, r15      ; encoding: [0x3f,0x40,A,A]

  mov #42, r15
  mov #42, 12(r15)
  mov #42, &disp
  mov disp, disp+2

; CHECK: mov #42, r15          ; encoding: [0x3f,0x40,0x2a,0x00]
; CHECK: mov #42, 12(r15)      ; encoding: [0xbf,0x40,0x2a,0x00,0x0c,0x00]
; CHECK: mov #42, &disp        ; encoding: [0xb2,0x40,0x2a,0x00,A,A]
; CHECK: mov disp, disp+2      ; encoding: [0x90,0x40,A,A,B,B]

  add r7, r8
  add 6(r7), r8
  add &disp, r8
  add disp, r8
  add @r9, r8
  add @r9+, r8
  add #42, r8

; CHECK: add r7, r8            ; encoding: [0x08,0x57]
; CHECK: add 6(r7), r8         ; encoding: [0x18,0x57,0x06,0x00]
; CHECK: add &disp, r8         ; encoding: [0x18,0x52,A,A]
; CHECK: add disp, r8          ; encoding: [0x18,0x50,A,A]
; CHECK: add @r9, r8           ; encoding: [0x28,0x59]
; CHECK: add @r9+, r8          ; encoding: [0x38,0x59]
; CHECK: add #42, r8           ; encoding: [0x38,0x50,0x2a,0x00]

  add r7, 6(r5)
  add 6(r7), 6(r5)
  add &disp, 6(r5)
  add disp, 6(r5)
  add @r9, 6(r5)
  add @r9+, 6(r5)
  add #42, 6(r5)

; CHECK: add r7, 6(r5)         ; encoding: [0x85,0x57,0x06,0x00]
; CHECK: add 6(r7), 6(r5)      ; encoding: [0x95,0x57,0x06,0x00,0x06,0x00]
; CHECK: add &disp, 6(r5)      ; encoding: [0x95,0x52,A,A,0x06,0x00]
; CHECK: add disp, 6(r5)       ; encoding: [0x95,0x50,A,A,0x06,0x00]
; CHECK: add @r9, 6(r5)        ; encoding: [0xa5,0x59,0x06,0x00]
; CHECK: add @r9+, 6(r5)       ; encoding: [0xb5,0x59,0x06,0x00]
; CHECK: add #42, 6(r5)        ; encoding: [0xb5,0x50,0x2a,0x00,0x06,0x00]

  add r7, &disp
  add 6(r7), &disp
  add &disp, &disp
  add disp, &disp
  add @r9, &disp
  add @r9+, &disp
  add #42, &disp

; CHECK: add r7, &disp         ; encoding: [0x82,0x57,A,A]
; CHECK: add 6(r7), &disp      ; encoding: [0x92,0x57,0x06,0x00,A,A]
; CHECK: add &disp, &disp      ; encoding: [0x92,0x52,A,A,B,B]
; CHECK: add disp, &disp       ; encoding: [0x92,0x50,A,A,B,B]
; CHECK: add @r9, &disp        ; encoding: [0xa2,0x59,A,A]
; CHECK: add @r9+, &disp       ; encoding: [0xb2,0x59,A,A]
; CHECK: add #42, &disp        ; encoding: [0xb2,0x50,0x2a,0x00,A,A]

  add r7, disp
  add 6(r7), disp
  add &disp, disp
  add disp, disp
  add @r9, disp
  add @r9+, disp
  add #42, disp

; CHECK: add r7, disp          ; encoding: [0x80,0x57,A,A]
; CHECK: add 6(r7), disp       ; encoding: [0x90,0x57,0x06,0x00,A,A]
; CHECK: add &disp, disp       ; encoding: [0x90,0x52,A,A,B,B]
; CHECK: add disp, disp        ; encoding: [0x90,0x50,A,A,B,B]
; CHECK: add @r9, disp         ; encoding: [0xa0,0x59,A,A]
; CHECK: add @r9+, disp        ; encoding: [0xb0,0x59,A,A]
; CHECK: add #42, disp         ; encoding: [0xb0,0x50,0x2a,0x00,A,A]

  call r7
  call 6(r7)
  call disp+6(r7)
  call &disp
  call disp
  call #disp

; CHECK: call r7               ; encoding: [0x87,0x12]
; CHECK: call 6(r7)            ; encoding: [0x97,0x12,0x06,0x00]
; CHECK: call disp+6(r7)       ; encoding: [0x97,0x12,A,A]
; CHECK: call &disp            ; encoding: [0x92,0x12,A,A]
; CHECK: call disp             ; encoding: [0x90,0x12,A,A]
; CHECK: call #disp            ; encoding: [0xb0,0x12,A,A]

disp:
  .word 0xcafe
  .word 0xbabe
