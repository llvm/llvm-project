; RUN: llc -enable-machine-outliner -verify-machineinstrs \
; RUN:     -mtriple=thumbv7m-unknown-none-eabihf < %s | FileCheck %s
; RUN: llc -enable-machine-outliner -verify-machineinstrs \
; RUN:     -mtriple=thumbv7m-unknown-none-eabihf \
; RUN:     --stop-after=machine-outliner < %s | FileCheck %s

; Verify that the Machine Outliner does not emit tTAILJMPr with a non-tcGPR register on Thumb. 

; CHECK-NOT: tTAILJMPr {{.*}}$r4

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-unknown-none-eabihf"

define i32 @__f_vfprintf(i1 %or.cond1640, i1 %tobool326.not) #0 {
entry:
  %0 = load ptr, ptr null, align 4
  br i1 %or.cond1640, label %if.then220, label %while.cond1098
if.then220:
  br i1 %tobool326.not, label %if.else391, label %if.then327
if.then327:
  %conv332 = select i1 false, i32 0, i32 0
  %call345 = call i32 %0(i8 0, ptr null)
  %call1217 = call i32 %0(i8 0, ptr null)
  ret i32 %call1217
if.else391:
  %call491 = call i32 %0(i8 0, ptr null)
  unreachable
while.cond1098:
  %call1104 = call i32 %0(i8 0, ptr null)
  br label %while.cond1098
}

attributes #0 = { minsize }