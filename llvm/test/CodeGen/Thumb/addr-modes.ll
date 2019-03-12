; REQUIRES: asserts
; RUN: llc < %s -debug-only=codegenprepare -o /dev/null 2>&1 | FileCheck %s

; These are regression tests for
;  https://bugs.llvm.org/show_bug.cgi?id=34106
;    "ARMTargetLowering::isLegalAddressingMode can accept incorrect
;    addressing modes for Thumb1 target"
;
; The Thumb1 target addressing modes don't support scaling.
; It supports: r1 + r2, where r1 and r2 can be the same register.

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-arm-none-eabi"

; Test case 01: %n is scaled by 4 (size of i32).
; Expected: GEP cannot be folded into LOAD.
; CHECK: local addrmode: [inbounds Base:%arrayidx]
define i32 @load01(i32* %p, i32 %n) nounwind {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 %n
  %0 = load i32, i32* %arrayidx, align 4
  ret i32 %0
}

; Test case 02: No scale of %n is needed because the size of i8 is 1.
; Expected: GEP can be folded into LOAD.
; CHECK: local addrmode: [inbounds Base:%p + 1*%n]
define i8 @load02(i8* %p, i32 %n) nounwind {
entry:
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %n
  %0 = load i8, i8* %arrayidx
  ret i8 %0
}

; Test case 03: 2*%x can be represented as %x + %x.
; Expected: GEP can be folded into LOAD.
; CHECK: local addrmode: [2*%x]
define i32 @load03(i32 %x) nounwind {
entry:
  %mul = shl nsw i32 %x, 1
  %0 = inttoptr i32 %mul to i32*
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

