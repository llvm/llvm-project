; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -O1 -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -ppc-gen-isel=false  < %s | FileCheck --check-prefix=CHECK-NO-ISEL %s
; RUN: llc -verify-machineinstrs -O1 -mcpu=pwr7 -ppc-gen-isel=false < %s | FileCheck --check-prefix=CHECK-NO-ISEL %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define zeroext i1 @testi1(i1 zeroext %b1, i1 zeroext %b2) #0 {
entry:
  %0 = tail call i8 asm "crand $0, $1, $2", "=^wc,^wc,^wc"(i1 %b1, i1 %b2) #0
  %1 = and i8 %0, 1
  %tobool3 = icmp ne i8 %1, 0
  ret i1 %tobool3

; CHECK-LABEL: @testi1
; CHECK-DAG: andi. {{[0-9]+}}, 3, 1
; CHECK-DAG: li [[REG1:[0-9]+]], 0
; CHECK-DAG: crmove [[REG2:[0-9]+]], 1
; CHECK-DAG: andi. {{[0-9]+}}, 4, 1
; CHECK-DAG: crand [[REG3:[0-9]+]], [[REG2]], 1
; CHECK-DAG: li [[REG4:[0-9]+]], 1
; CHECK: isel 3, [[REG4]], [[REG1]], [[REG3]]
; CHECK-NO-ISEL-LABEL: @testi1
; CHECK-NO-ISEL: bc 12, 20,
; CHECK-NO-ISEL: blr
; CHECK-NO-ISEL: addi 3, 4, 0
; CHECK-NO-ISEL-NEXT: blr
; CHECK: blr
}

define signext i32 @testi32(i32 signext %b1, i32 signext %b2) #0 {
entry:
  %0 = tail call i32 asm "crand $0, $1, $2", "=^wc,^wc,^wc"(i32 %b1, i32 %b2) #0
  ret i32 %0

; The ABI sign_extend should combine with the any_extend from the asm result,
; and the result will be 0 or -1. This highlights the fact that only the first
; bit is meaningful.
; CHECK-LABEL: @testi32
; CHECK-DAG: andi. {{[0-9]+}}, 3, 1
; CHECK-DAG: li [[REG1:[0-9]+]], 0
; CHECK-DAG: crmove [[REG2:[0-9]+]], 1
; CHECK-DAG: andi. {{[0-9]+}}, 4, 1
; CHECK-DAG: crand [[REG3:[0-9]+]], [[REG2]], 1
; CHECK-DAG: li [[REG4:[0-9]+]], -1
; CHECK: isel 3, [[REG4]], [[REG1]], [[REG3]]
; CHECK: blr
}

define zeroext i8 @testi8(i8 zeroext %b1, i8 zeroext %b2) #0 {
entry:
  %0 = tail call i8 asm "crand $0, $1, $2", "=^wc,^wc,^wc"(i8 %b1, i8 %b2) #0
  ret i8 %0

; CHECK-LABEL: @testi8
; CHECK-DAG: andi. {{[0-9]+}}, 3, 1
; CHECK-DAG: li [[REG1:[0-9]+]], 0
; CHECK-DAG: crmove [[REG2:[0-9]+]], 1
; CHECK-DAG: andi. {{[0-9]+}}, 4, 1
; CHECK-DAG: crand [[REG3:[0-9]+]], [[REG2]], 1
; CHECK-DAG: li [[REG4:[0-9]+]], 1
; CHECK: isel 3, [[REG4]], [[REG1]], [[REG3]]
; CHECK: blr
}

attributes #0 = { nounwind "target-features"="+crbits" }

