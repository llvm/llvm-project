; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-crbits | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck --check-prefix=CHECK-CRB %s
; RUN: llc -verify-machineinstrs -ppc-gen-isel=false < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck --check-prefix=CHECK-NO-ISEL %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @test1(i1 %a, i32 %c) nounwind  {
  %x = select i1 %a, i32 %c, i32 0
  ret i32 %x

; CHECK-LABEL: @test1
; CHECK-NOT: li {{[0-9]+}}, 0
; CHECK: iseleq 3, 0,
; CHECK: blr
; CHECK-NO-ISEL-LABEL: @test1
; CHECK-NO-ISEL: li 3, 0
; CHECK-NO-ISEL-NEXT: bc 12, 1, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL-NEXT: blr
; CHECK-NO-ISEL-NEXT: [[TRUE]]
; CHECK-NO-ISEL-NEXT: addi 3, 4, 0
; CHECK-NO-ISEL-NEXT: blr
}

define i32 @test2(i1 %a, i32 %c) nounwind  {
  %x = select i1 %a, i32 0, i32 %c
  ret i32 %x

; CHECK-CRB-LABEL: @test2
; CHECK-CRB-NOT: li {{[0-9]+}}, 0
; CHECK-CRB: iselgt 3, 0,
; CHECK-CRB: blr
; CHECK-NO-ISEL-LABEL: @test2
; CHECK-NO-ISEL: bc 12, 1, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL: ori 3, 4, 0
; CHECK-NO-ISEL-NEXT: blr
; CHECK-NO-ISEL-NEXT: [[TRUE]]
; CHECK-NO-ISEL-NEXT: li 3, 0
; CHECK-NO-ISEL-NEXT: blr
}

