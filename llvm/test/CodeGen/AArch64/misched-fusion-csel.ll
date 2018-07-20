; RUN: llc %s -o - -mtriple=aarch64-unknown -mattr=fuse-csel | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=exynos-m3  | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=exynos-m4  | FileCheck %s

target triple = "aarch64-unknown"

define i32 @test_sub_cselw(i32 %a0, i32 %a1, i32 %a2) {
entry:
  %v0 = sub i32 %a0, 13
  %cond = icmp eq i32 %v0, 0
  %v1 = add i32 %a1, 7
  %v2 = select i1 %cond, i32 %a0, i32 %v1
  ret i32 %v2

; CHECK-LABEL: test_sub_cselw:
; CHECK: cmp {{w[0-9]}}, #13
; CHECK-NEXT: csel {{w[0-9]}}
}

define i64 @test_sub_cselx(i64 %a0, i64 %a1, i64 %a2) {
entry:
  %v0 = sub i64 %a0, 13
  %cond = icmp eq i64 %v0, 0
  %v1 = add i64 %a1, 7
  %v2 = select i1 %cond, i64 %a0, i64 %v1
  ret i64 %v2

; CHECK-LABEL: test_sub_cselx:
; CHECK: cmp {{x[0-9]}}, #13
; CHECK-NEXT: csel {{x[0-9]}}
}
