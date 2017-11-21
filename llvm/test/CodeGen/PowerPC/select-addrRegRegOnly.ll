; RUN: llc -mcpu=pwr8 -mtriple=powerpc64le-unknown-unknown -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mcpu=pwr8 -mtriple=powerpc64-unknown-unknown -verify-machineinstrs < %s | FileCheck %s

; Function Attrs: norecurse nounwind readonly
define float @testSingleAccess(i32* nocapture readonly %arr) local_unnamed_addr #0 {
; CHECK-LABEL: testSingleAccess:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    addi 3, 3, 8
; CHECK-NEXT:    lfiwax 0, 0, 3
; CHECK-NEXT:    xscvsxdsp 1, 0
; CHECK-NEXT:    blr
entry:
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 2
  %0 = load i32, i32* %arrayidx, align 4
  %conv = sitofp i32 %0 to float
  ret float %conv
}

; Function Attrs: norecurse nounwind readonly
define float @testMultipleAccess(i32* nocapture readonly %arr) local_unnamed_addr #0 {
; CHECK-LABEL: testMultipleAccess:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    lwz 4, 8(3)
; CHECK-NEXT:    lwz 12, 12(3)
; CHECK-NEXT:    add 3, 12, 4
; CHECK-NEXT:    mtvsrwa 0, 3
; CHECK-NEXT:    xscvsxdsp 1, 0
; CHECK-NEXT:    blr
entry:
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 2
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %arr, i64 3
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %conv = sitofp i32 %add to float
  ret float %conv
}
