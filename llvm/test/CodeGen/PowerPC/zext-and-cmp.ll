; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu  < %s | FileCheck %s

; Test that we recognize that an 'and' instruction that feeds a comparison
; to zero can be simplifed by using the record form when one of its operands
; is known to be zero extended.

@k = local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind
define signext i32 @cmplwi(i32* nocapture readonly %p, i32* nocapture readonly %q, i32 signext %j, i32 signext %r10) {
entry:
  %0 = load i32, i32* %q, align 4
  %shl = shl i32 %0, %j
  %1 = load i32, i32* %p, align 4
  %and = and i32 %shl, %r10
  %and1 = and i32 %and, %1
  %tobool = icmp eq i32 %and1, 0
  br i1 %tobool, label %cleanup, label %if.then

if.then:
  store i32 %j, i32* @k, align 4
  br label %cleanup

cleanup:
  %retval.0 = phi i32 [ 0, %if.then ], [ 1, %entry ]
  ret i32 %retval.0
}

; CHECK-LABEL: cmplwi:
; CHECK:      lwz [[T1:[0-9]+]], 0(3)
; CHECK:      and. {{[0-9]+}}, {{[0-9]+}}, [[T1]]
; CHECK-NOT:  cmplwi
; CHECK-NEXT: beq      0,
