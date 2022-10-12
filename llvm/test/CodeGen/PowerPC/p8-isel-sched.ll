; RUN: llc -verify-machineinstrs -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -ppc-gen-isel=false < %s | FileCheck --check-prefix=CHECK-NO-ISEL %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo(ptr nocapture %r1, ptr nocapture %r2, ptr nocapture %r3, ptr nocapture %r4, i32 signext %a, i32 signext %b, i32 signext %c, i32 signext %d) #0 {
entry:
  %tobool = icmp ne i32 %a, 0
  %cond = select i1 %tobool, i32 %b, i32 %c
  store i32 %cond, ptr %r1, align 4
  %cond5 = select i1 %tobool, i32 %b, i32 %d
  store i32 %cond5, ptr %r2, align 4
  %add = add nsw i32 %b, 1
  %sub = add nsw i32 %d, -2
  %cond10 = select i1 %tobool, i32 %add, i32 %sub
  store i32 %cond10, ptr %r3, align 4
  %add13 = add nsw i32 %b, 3
  %sub15 = add nsw i32 %d, -5
  %cond17 = select i1 %tobool, i32 %add13, i32 %sub15
  store i32 %cond17, ptr %r4, align 4
  ret void
}

; Make sure that we don't schedule all of the isels together, they should be
; intermixed with the adds because each isel starts a new dispatch group.
; CHECK-LABEL: @foo
; CHECK-NO-ISEL-LABEL: @foo
; CHECK: isel
; CHECK-NO-ISEL: bc 12, 2, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK-NO-ISEL: [[TRUE]]
; CHECK-NO-ISEL-NEXT: addi {{[0-9]+}}, {{[0-9]+}}, 0
; CHECK: addi
; CHECK: isel
; CHECK-NO-ISEL: bc 12, 2, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL: ori 10, 11, 0
; CHECK-NO-ISEL-NEXT: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK-NO-ISEL: [[TRUE]]
; CHECK: blr

attributes #0 = { nounwind }
