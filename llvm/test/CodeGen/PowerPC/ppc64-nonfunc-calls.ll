; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.cd = type { i64, i64, i64 }

@something = global [33 x i8] c"this is not really code, but...\0A\00", align 1
@tls_something = thread_local global %struct.cd zeroinitializer, align 8
@extern_something = external global %struct.cd

; Function Attrs: nounwind
define void @foo() #0 {
entry:
  tail call void bitcast ([33 x i8]* @something to void ()*)() #0
  ret void

; CHECK-LABEL: @foo
; CHECK-DAG: addis [[REG1:[0-9]+]], 2, something@toc@ha
; CHECK-DAG: std 2, 40(1)
; CHECK-DAG: addi [[REG3:[0-9]+]], [[REG1]], something@toc@l
; CHECK-DAG: ld [[REG2:[0-9]+]], 0([[REG3]])
; CHECK-DAG: ld 11, 16([[REG3]])
; CHECK-DAG: ld 2, 8([[REG3]])
; CHECK-DAG: mtctr [[REG2]]
; CHECK: bctrl
; CHECK: ld 2, 40(1)
; CHECK: blr
}

; Function Attrs: nounwind
define void @bar() #0 {
entry:
  tail call void bitcast (%struct.cd* @tls_something to void ()*)() #0
  ret void

; CHECK-LABEL: @bar
; CHECK-DAG: addis [[REG1:[0-9]+]], 13, tls_something@tprel@ha
; CHECK-DAG: std 2, 40(1)
; CHECK-DAG: addi [[REG3:[0-9]+]], [[REG1]], tls_something@tprel@l
; CHECK-DAG: ld [[REG2:[0-9]+]], 0([[REG3]])
; CHECK-DAG: ld 11, 16([[REG3]])
; CHECK-DAG: ld 2, 8([[REG3]])
; CHECK-DAG: mtctr [[REG2]]
; CHECK: bctrl
; CHECK: ld 2, 40(1)
; CHECK: blr
}

; Function Attrs: nounwind
define void @ext() #0 {
entry:
  tail call void bitcast (%struct.cd* @extern_something to void ()*)() #0
  ret void

; CHECK-LABEL: @ext
; CHECK-DAG: addis [[REG1:[0-9]+]], 2, [[NAME:[._A-Za-z0-9]+]]@toc@ha
; CHECK-DAG: std 2, 40(1)
; CHECK-DAG: ld [[REG3:[0-9]+]], [[NAME]]@toc@l(3)
; CHECK-DAG: ld [[REG2:[0-9]+]], 0([[REG3]])
; CHECK-DAG: ld 11, 16([[REG3]])
; CHECK-DAG: ld 2, 8([[REG3]])
; CHECK-DAG: mtctr [[REG2]]
; CHECK: bctrl
; CHECK: ld 2, 40(1)
; CHECK: blr
}

attributes #0 = { nounwind }

