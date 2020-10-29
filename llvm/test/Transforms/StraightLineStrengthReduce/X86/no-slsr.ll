; RUN: opt < %s -slsr -gvn -S | FileCheck %s
; RUN: opt < %s -passes='slsr,gvn' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Do not perform SLSR on &input[s] and &input[s * 2] which fit into addressing
; modes of X86.
define i32 @no_slsr_gep(i32* %input, i64 %s) {
; CHECK-LABEL: @no_slsr_gep(
  ; v0 = input[0];
  %p0 = getelementptr inbounds i32, i32* %input, i64 0
  %v0 = load i32, i32* %p0

  ; v1 = input[s];
  %p1 = getelementptr inbounds i32, i32* %input, i64 %s
; CHECK: %p1 = getelementptr inbounds i32, i32* %input, i64 %s
  %v1 = load i32, i32* %p1

  ; v2 = input[s * 2];
  %s2 = mul nsw i64 %s, 2
  %p2 = getelementptr inbounds i32, i32* %input, i64 %s2
; CHECK: %p2 = getelementptr inbounds i32, i32* %input, i64 %s2
  %v2 = load i32, i32* %p2

  ; return v0 + v1 + v2;
  %1 = add i32 %v0, %v1
  %2 = add i32 %1, %v2
  ret i32 %2
}

define void @no_slsr_add(i32 %b, i32 %s) {
; CHECK-LABEL: @no_slsr_add(
  %1 = add i32 %b, %s
; CHECK: add i32 %b, %s
  call void @foo(i32 %1)
  %s2 = mul i32 %s, 2
; CHECK: %s2 = mul i32 %s, 2
  %2 = add i32 %b, %s2
; CHECK: add i32 %b, %s2
  call void @foo(i32 %2)
  ret void
}

declare void @foo(i32 %a)
