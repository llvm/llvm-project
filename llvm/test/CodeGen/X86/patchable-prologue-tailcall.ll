; RUN: llc -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK

; CHECK: "?mi_new_test@@YAPEAX_K@Z":
; CHECK-NEXT: # %bb.0:
; CHECK-NEXT: jmp     mi_new                          # TAILCALL

; CHECK: "?builtin_malloc_test@@YAPEAX_K@Z":
; CHECK-NEXT: # %bb.0:
; CHECK-NEXT: jmp     malloc                          # TAILCALL

; // Built with: clang-cl.exe /c patchable-prologue-tailcall.cpp /O2 /hotpatch -Xclang -emit-llvm
;
; typedef unsigned long long size_t;
; 
; extern "C" {
;     void* mi_new(size_t size);
;  }
;
; void *mi_new_test(size_t count)
; {
;     return mi_new(count);
; }
; 
; void *builtin_malloc_test(size_t count)
; {
;     return __builtin_malloc(count);
; }

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.38.33133"

define dso_local noundef ptr @"?mi_new_test@@YAPEAX_K@Z"(i64 noundef %count) local_unnamed_addr "patchable-function"="prologue-short-redirect" {
entry:
  %call = tail call ptr @mi_new(i64 noundef %count)
  ret ptr %call
}

declare dso_local ptr @mi_new(i64 noundef) local_unnamed_addr

define dso_local noalias noundef ptr @"?builtin_malloc_test@@YAPEAX_K@Z"(i64 noundef %count) local_unnamed_addr "patchable-function"="prologue-short-redirect" {
entry:
  %call = tail call ptr @malloc(i64 noundef %count)
  ret ptr %call
}

declare dso_local noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #0

attributes #0 = { allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{i32 1, !"MaxTLSAlign", i32 65536}
