; Allocas in non-entry blocks are hoisted to the entry block, alloca i64
; sizes are normalized to i32, and `disjoint` flags are stripped.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK: entry:
; CHECK: %hoisted = alloca i32
; CHECK: %a = alloca i32, i32 4
; CHECK-NOT: disjoint

define void @kernel(ptr addrspace(1) %out, i32 %n) {
entry:
  %a = alloca i32, i64 4
  %or = or disjoint i32 %n, 1
  br label %body

body:
  %hoisted = alloca i32
  store i32 %or, ptr %hoisted
  ret void
}
