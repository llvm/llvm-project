; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -ppc-gen-isel=false < %s | FileCheck --check-prefix=CHECK-NO-ISEL %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define signext i32 @crbitsoff(i32 signext %v1, i32 signext %v2) #0 {
entry:
  %tobool = icmp ne i32 %v1, 0
  %lnot = icmp eq i32 %v2, 0
  %and3 = and i1 %tobool, %lnot
  %and = zext i1 %and3 to i32
  ret i32 %and

; CHECK-LABEL: @crbitsoff
; CHECK-NO-ISEL-LABEL: @crbitsoff
; CHECK-DAG: cmplwi {{[0-9]+}}, 3, 0
; CHECK-DAG: li [[REG2:[0-9]+]], 1
; CHECK-DAG: cntlzw [[REG3:[0-9]+]],
; CHECK: isel [[REG4:[0-9]+]], 0, [[REG2]]
; CHECK-NO-ISEL: bc 12, 2, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL: ori 4, 5, 0
; CHECK-NO-ISEL-NEXT: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK-NO-ISEL: [[TRUE]]
; CHECK-NO-ISEL-NEXT: addi 4, 0, 0
; CHECK: and 3, [[REG4]], [[REG3]]
; CHECK: blr
}

define signext i32 @crbitson(i32 signext %v1, i32 signext %v2) #1 {
entry:
  %tobool = icmp ne i32 %v1, 0
  %lnot = icmp eq i32 %v2, 0
  %and3 = and i1 %tobool, %lnot
  %and = zext i1 %and3 to i32
  ret i32 %and

; CHECK-LABEL: @crbitson
; CHECK-NO-ISEL-LABEL: @crbitson
; CHECK-DAG: cmpwi {{[0-9]+}}, 3, 0
; CHECK-DAG: cmpwi {{[0-9]+}}, 4, 0
; CHECK-DAG: li [[REG2:[0-9]+]], 1
; CHECK-DAG: crorc [[REG3:[0-9]+]],
; CHECK: isel 3, 0, [[REG2]], [[REG3]]
; CHECK-NO-ISEL: bc 12, 20, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL-NEXT: blr
; CHECK-NO-ISEL: [[TRUE]]
; CHECK-NO-ISEL-NEXT: addi 3, 0, 0
; CHECK-NO-ISEL-NEXT: blr
; CHECK: blr
}


attributes #0 = { nounwind readnone "target-features"="-crbits" }
attributes #1 = { nounwind readnone }

