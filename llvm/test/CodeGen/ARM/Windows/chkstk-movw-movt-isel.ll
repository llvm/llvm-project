; RUN: llc -mtriple thumbv7--windows-itanium -code-model large -verify-machineinstrs -filetype obj -o - %s \
; RUN:    | llvm-objdump --no-show-raw-insn -d - | FileCheck %s

; ModuleID = 'reduced.c'
target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7--windows-itanium"

define arm_aapcs_vfpcc i8 @isel(i32 %i) {
entry:
  %i.addr = alloca i32, align 4
  %buffer = alloca [4096 x i8], align 1
  store i32 %i, ptr %i.addr, align 4
  %0 = load i32, ptr %i.addr, align 4
  %rem = urem i32 %0, 4096
  %arrayidx = getelementptr inbounds [4096 x i8], ptr %buffer, i32 0, i32 %rem
  %1 = load volatile i8, ptr %arrayidx, align 1
  ret i8 %1
}

; CHECK-LABEL: isel
; CHECK: push {r4, r5, r6, lr}
; CHECK-DAG: movw r12, #0
; CHECK-DAG: movt r12, #0
; CHECK-DAG: movw r4, #{{\d*}}
; CHECK: blx r12
; CHECK: sub.w sp, sp, r4

