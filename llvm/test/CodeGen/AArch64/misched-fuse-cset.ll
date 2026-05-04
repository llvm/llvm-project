; RUN: llc %s -o - -mtriple=aarch64-unknown -mattr=+fuse-cset | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a78  | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a710  | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a715  | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a720  | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a725  | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-x4  | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-x925  | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=neoverse-n2  | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=neoverse-n3  | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=neoverse-v1  | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=neoverse-v2  | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=neoverse-v3  | FileCheck %s

target triple = "aarch64-unknown"

define i32 @test_sub_cselw(i32 %a0, i32 %a1, i32 %a2) {
entry:
  %v0 = sub i32 %a0, 13
  %cond = icmp eq i32 %v0, 0
  %v1 = add i32 %a1, 7
  %v2 = select i1 %cond, i32 0, i32 1
  %v3 = xor i32 %v1, %v2
  ret i32 %v3

; CHECK-LABEL: test_sub_cselw:
; CHECK: cmp {{w[0-9]}}, #13
; CHECK-NEXT: cset {{w[0-9]}}
}

define i64 @test_sub_cselx(i64 %a0, i64 %a1, i64 %a2) {
entry:
  %v0 = sub i64 %a0, 13
  %cond = icmp eq i64 %v0, 0
  %v1 = add i64 %a1, 7
  %v2 = select i1 %cond, i64 0, i64 1
  %v3 = xor i64 %v1, %v2
  ret i64 %v3

; CHECK-LABEL: test_sub_cselx:
; CHECK: cmp {{x[0-9]}}, #13
; CHECK-NEXT: cset {{w[0-9]}}
}
