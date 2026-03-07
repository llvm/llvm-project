; RUN: llc -mtriple=bpf -mattr=+has-i128-direct-return < %s | FileCheck %s

; Source code:
; typedef struct {
;     long long v[2];
; } arr2_i64;
; 
; arr2_i64 bar(int a, int b, int c, int d, int e);
; 
; arr2_i64 foo(int a, int b, int c) {
;     return bar(a, b, c, 1, 2);
; }
; 
; Compile with:
; 	clang -target bpf -O2 -S -emit-llvm foo.c

; Function Attrs: nounwind uwtable
define [2 x i64] @foo(i32 %a, i32 %b, i32 %c) #0 {
; CHECK-LABEL: foo:
; CHECK: w4 = 1
; CHECK-NEXT: w5 = 2
entry:
  %call = tail call [2 x i64]  @bar(i32 %a, i32 %b, i32 %c, i32 1, i32 2) #3
  ret [2 x i64] %call
}

declare [2 x i64] @bar(i32, i32, i32, i32, i32) #1
