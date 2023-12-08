; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-elf"

@g_29 = common global i8 0, align 1               ; <i8*> [#uses=0]

define signext i8 @foo(i8 signext %_si1, i8 signext %_si2) nounwind readnone {
entry:
; CHECK-LABEL: foo:
; CHECK: call #__mspabi_mpyi
  %mul = mul i8 %_si2, %_si1                      ; <i8> [#uses=1]
  ret i8 %mul
}

define void @uint81(i16* nocapture %p_32) nounwind {
entry:
  %call = tail call i16 @bar(i8* bitcast (i8 (i8, i8)* @foo to i8*)) nounwind ; <i16> [#uses=0]
  ret void
}

declare i16 @bar(i8*)
