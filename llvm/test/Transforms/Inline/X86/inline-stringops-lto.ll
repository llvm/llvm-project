; REQUIRES: x86-registered-target
; RUN: split-file %s %t
; RUN: llvm-link %t/mod_a.ll %t/mod_b.ll -S -o %t/linked.ll
; RUN: opt -passes='default<O2>' %t/linked.ll -S -o %t/opt.ll
; RUN: llc -mtriple=x86_64-unknown-linux-gnu %t/opt.ll -o - | FileCheck %s

; Inlined opted-in code across modules must not produce a memset call.
; CHECK-LABEL: caller_inlines_optin:
; CHECK-NOT:   callq memset
; CHECK-NOT:   jmp memset
; CHECK:       retq

; Opted-in callee non-inlined call.
; CHECK-LABEL: caller_calls_optin_noinline:
; CHECK:       jmp optin_noinline

; Unrelated unmarked function retains standard lowering to memset.
; CHECK-LABEL: unmarked_function:
; CHECK:       jmp memset@PLT

;--- mod_a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @optin_callee(ptr %p) #0 {
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 512, i1 false)
  ret void
}

define void @optin_noinline(ptr %p) noinline #0 {
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 512, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

attributes #0 = { "inline-memset"="true" }

;--- mod_b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @optin_callee(ptr)
declare void @optin_noinline(ptr)

define void @caller_inlines_optin(ptr %p) {
entry:
  call void @optin_callee(ptr %p)
  ret void
}

define void @caller_calls_optin_noinline(ptr %p) {
entry:
  call void @optin_noinline(ptr %p)
  ret void
}

define void @unmarked_function(ptr %p) {
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 512, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
