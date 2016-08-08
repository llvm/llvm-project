; RUN: llc -verify-machineinstrs -mcpu=g5 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @_ZNK4llvm5APInt17countLeadingZerosEv(i64 *%t) nounwind {
        %tmp19 = load i64, i64* %t
        %tmp22 = tail call i64 @llvm.ctlz.i64( i64 %tmp19, i1 true )             ; <i64> [#uses=1]
        %tmp23 = trunc i64 %tmp22 to i32
        %tmp89 = add i32 %tmp23, -64          ; <i32> [#uses=1]
        %tmp90 = add i32 %tmp89, 0            ; <i32> [#uses=1]
        ret i32 %tmp90

; CHECK-LABEL: @_ZNK4llvm5APInt17countLeadingZerosEv
; CHECK: ld [[REG1:[0-9]+]], 0(3)
; CHECK-NEXT: cntlzd [[REG2:[0-9]+]], [[REG1]]
; CHECK-NEXT: addi 3, [[REG2]], -64
; CHECK-NEXT: blr
}

declare i64 @llvm.ctlz.i64(i64, i1)
