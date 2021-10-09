; RUN: llc -o - %s | FileCheck %s

; AsmPrinter cannot lower floating point constant expressions in global
; initializers. Check that we do not create new globals with float constant
; expressions in initializers.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios14.0.0"

define [1 x <4 x float>] @test1() {
; CHECK-LABEL:    .p2align    4               ; -- Begin function test1
; CHECK-NEXT: lCPI0_0:
; CHECK-NEXT:     .quad   0                       ; 0x0
; CHECK-NEXT:     .quad   4575657221408423936     ; 0x3f80000000000000
; CHECK-NEXT:     .section    __TEXT,__text,regular,pure_instructions
; CHECK-NEXT:     .globl  _test1
; CHECK-NEXT:     .p2align    2
; CHECK-NEXT: _test1:                                 ; @test1
; CHECK-NEXT:     .cfi_startproc
; CHECK-NEXT: ; %bb.0:
; CHECK-NEXT: Lloh0:
; CHECK-NEXT:     adrp    x8, lCPI0_0@PAGE
; CHECK-NEXT: Lloh1:
; CHECK-NEXT:     ldr q0, [x8, lCPI0_0@PAGEOFF]
; CHECK-NEXT:     ret

  ret [1 x <4 x float>] [<4 x float> bitcast (<1 x i128> <i128 84405977732342157929391748327801880576> to <4 x float>)]
}

define [1 x <4 x float>] @test2() {
; CHECK-LABEL:    .p2align    4               ; -- Begin function test2
; CHECK-NEXT: lCPI1_0:
; CHECK-NEXT:     .long   0x00000000              ; float 0
; CHECK-NEXT:     .long   0x00000000              ; float 0
; CHECK-NEXT:     .long   0x00000000              ; float 0
; CHECK-NEXT:     .long   0x3f800000              ; float 1
; CHECK-NEXT:     .section    __TEXT,__text,regular,pure_instructions
; CHECK-NEXT:     .globl  _test2
; CHECK-NEXT:     .p2align    2
; CHECK-NEXT: _test2:                                 ; @test2
; CHECK-NEXT:     .cfi_startproc
; CHECK-NEXT: ; %bb.0:
; CHECK-NEXT: Lloh2:
; CHECK-NEXT:     adrp    x8, lCPI1_0@PAGE
; CHECK-NEXT: Lloh3:
; CHECK-NEXT:     ldr q1, [x8, lCPI1_0@PAGEOFF]
; CHECK-NEXT:     mov s2, v1[1]
; CHECK-NEXT:     fneg    s0, s1
; CHECK-NEXT:     mov s3, v1[2]
; CHECK-NEXT:     mov s1, v1[3]
; CHECK-NEXT:     fneg    s2, s2
; CHECK-NEXT:     fneg    s1, s1
; CHECK-NEXT:     mov.s   v0[1], v2[0]
; CHECK-NEXT:     fneg    s2, s3
; CHECK-NEXT:     mov.s   v0[2], v2[0]
; CHECK-NEXT:     mov.s   v0[3], v1[0]
; CHECK-NEXT:     ret
;
  ret [1 x <4 x float>] [<4 x float>
    <float fneg (float extractelement (<4 x float> bitcast (<1 x i128> <i128 84405977732342157929391748327801880576> to <4 x float>), i32 0)),
     float fneg (float extractelement (<4 x float> bitcast (<1 x i128> <i128 84405977732342157929391748327801880576> to <4 x float>), i32 1)),
     float fneg (float extractelement (<4 x float> bitcast (<1 x i128> <i128 84405977732342157929391748327801880576> to <4 x float>), i32 2)),
     float fneg (float extractelement (<4 x float> bitcast (<1 x i128> <i128 84405977732342157929391748327801880576> to <4 x float>), i32 3))>]
}
