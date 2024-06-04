; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mattr=cmp-bcc-fusion | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=cortex-a77    | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=cortex-a78    | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=cortex-a78ae  | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=cortex-a78c   | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=cortex-a710   | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=cortex-x715   | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=cortex-x720   | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=cortex-x720ae | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=cortex-x1     | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=cortex-x2     | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=neoverse-v2   | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=ampere1       | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=ampere1a      | FileCheck %s
; RUN: llc %s -o - -O0 -mtriple=aarch64-unknown -mcpu=ampere1b      | FileCheck %s


define void @test_cmp_bcc_fusion(i32 %x, i32 %y, i32* %arr) {
entry:
  %cmp = icmp eq i32 %x, %y
  store i32 %x, i32* %arr, align 4
  br i1 %cmp, label %if_true, label %if_false

if_true:
  ret void

if_false:
  ret void
}

; CHECK-LABEL: test_cmp_bcc_fusion:
; CHECK: str {{w[0-9]}}, [{{x[0-9]}}]
; CHECK-NEXT: subs {{w[0-9]}}, {{w[0-9]}}, {{w[0-9]}}
; CHECK-NEXT: b.ne .LBB0_2
; CHECK-NEXT: b .LBB0_1
