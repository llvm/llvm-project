; REQUIRES: arm-registered-target
; RUN: opt -module-summary -o %t %s
; RUN: llvm-bcanalyzer -dump %t | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7--linux-android"

module asm "asm:"
module asm "bx lr"

; NotEligibleToImport
; CHECK: <PERMODULE {{.*}} op1=16
define void @f() {
  call void asm sideeffect "bl asm\0A", ""()
  ret void
}
