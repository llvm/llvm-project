; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | FileCheck %s

; The tests check the behavior of the tail call decision when the callee is speculatable.

; Callee should be tail called in this function since it is at a tail call position.
define dso_local double @speculatable_callee_return_use_only (double* nocapture %res, double %a) #0 {
; CHECK-LABEL: speculatable_callee_return_use_only:
; CHECK: # %bb.0: # %entry
; CHECK-NEXT: b callee
entry:
  %value = tail call double @callee(double %a) #2
  ret double %value
}

; Callee should not be tail called since it is not at a tail call position.
define dso_local void @speculatable_callee_non_return_use_only (double* nocapture %res, double %a) #0 {
; CHECK-LABEL: speculatable_callee_non_return_use_only:
; CHECK: # %bb.0: # %entry
; CHECK-NEXT: mflr r0
; CHECK-NEXT: std r30, -16(r1)  # 8-byte Folded Spill
; CHECK-NEXT: std r0, 16(r1)
; CHECK-NEXT: stdu r1, -48(r1)
; CHECK-NEXT: mr r30, r3
; CHECK-NEXT: bl callee
; CHECK-NEXT: stfdx f1, 0, r30
; CHECK-NEXT: addi r1, r1, 48
; CHECK-NEXT: ld r0, 16(r1)
; CHECK-NEXT: ld r30, -16(r1) # 8-byte Folded Reload
; CHECK-NEXT: mtlr r0
; CHECK-NEXT: blr
entry:
  %call = tail call double @callee(double %a) #2
  store double %call, double* %res, align 8
  ret void
}

; Callee should not be tail called since it is not at a tail call position.
define dso_local double @speculatable_callee_multi_use (double* nocapture %res, double %a) #0 {
  ; CHECK-LABEL: speculatable_callee_multi_use:
  ; CHECK: # %bb.0: # %entry
  ; CHECK-NEXT: mflr r0
  ; CHECK-NEXT: std r30, -16(r1)  # 8-byte Folded Spill
  ; CHECK-NEXT: std r0, 16(r1)
  ; CHECK-NEXT: stdu r1, -48(r1)
  ; CHECK-NEXT: mr r30, r3
  ; CHECK-NEXT: bl callee
  ; CHECK-NEXT: stfdx f1, 0, r30
  ; CHECK-NEXT: addi r1, r1, 48
  ; CHECK-NEXT: ld r0, 16(r1)
  ; CHECK-NEXT: ld r30, -16(r1) # 8-byte Folded Reload
  ; CHECK-NEXT: mtlr r0
  ; CHECK-NEXT: blr
  entry:
  %call = tail call double @callee(double %a) #2
  store double %call, double* %res, align 8
  ret double %call
}

; Callee should not be tail called since it is not at a tail call position.
; FIXME: A speculatable callee can be tail called if it is moved into a valid tail call position.
define dso_local double @speculatable_callee_intermediate_instructions (double* nocapture %res, double %a) #0 {
  ; CHECK-LABEL: speculatable_callee_intermediate_instructions:
  ; CHECK: # %bb.0: # %entry
  ; CHECK-NEXT: mflr r0
  ; CHECK-NEXT: std r30, -16(r1)  # 8-byte Folded Spill
  ; CHECK-NEXT: std r0, 16(r1)
  ; CHECK-NEXT: stdu r1, -48(r1)
  ; CHECK-NEXT: mr r30, r3
  ; CHECK-NEXT: bl callee
  ; CHECK-NEXT: lis r3, 16404
  ; CHECK-NEXT: ori r3, r3, 52428
  ; CHECK-NEXT: sldi r3, r3, 32
  ; CHECK-NEXT: oris r3, r3, 52428
  ; CHECK-NEXT: ori r3, r3, 52429
  ; CHECK-NEXT: std r3, 0(r30)
  ; CHECK-NEXT: addi r1, r1, 48
  ; CHECK-NEXT: ld r0, 16(r1)
  ; CHECK-NEXT: ld r30, -16(r1)  # 8-byte Folded Reload
  ; CHECK-NEXT: mtlr r0
  ; CHECK-NEXT: blr

  entry:
  %call = tail call double @callee(double %a) #2
  store double 5.2, double* %res, align 8
  ret double %call
}


define double @callee(double) #1 {
  ret double 4.5
}

attributes #0 = { nounwind }
attributes #1 = { readnone speculatable }
attributes #2 = { nounwind noinline }
