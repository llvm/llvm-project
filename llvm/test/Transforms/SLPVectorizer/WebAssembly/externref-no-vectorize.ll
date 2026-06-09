; RUN: opt < %s -passes=slp-vectorizer -mtriple=wasm32-unknown-unknown -S | FileCheck %s

; A WebAssembly reference type cannot be vectorized.
; This used to crash type legalization while the SLP vectorizer was computing
; the cost for a <N x externref> value.

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

%externref = type ptr addrspace(10)

declare %externref @foo()
declare void @bar(%externref)

; Two externref phis that feed two calls are candidates for being gathered into
; a vector but they must be left scalar. Derived the following C code:
;
;   __externref_t foo(void);
;   void bar(__externref_t);
;   void test(int flag, __externref_t ref1, __externref_t ref2) {
;     if (flag) {
;       ref1 = foo();
;       ref2 = foo();
;     }
;     bar(ref1); bar(ref2);
;   }
;
; CHECK-LABEL: @test(
; CHECK-NOT:   phi <{{.*}} x ptr addrspace(10)>
; CHECK-NOT:   call void @bar(<{{.*}} x ptr addrspace(10)>
; CHECK:       ret void
define void @test(i32 %flag, %externref %ref1, %externref %ref2) {
entry:
  %c = icmp eq i32 %flag, 0
  br i1 %c, label %join, label %then

then:
  %a = tail call %externref @foo()
  %b = tail call %externref @foo()
  br label %join

join:
  %p1 = phi %externref [ %a, %then ], [ %ref1, %entry ]
  %p2 = phi %externref [ %b, %then ], [ %ref2, %entry ]
  tail call void @bar(%externref %p1)
  tail call void @bar(%externref %p2)
  ret void
}
