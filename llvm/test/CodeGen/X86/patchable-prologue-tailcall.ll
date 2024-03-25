; RUN: llc -verify-machineinstrs -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=CHECK

; CHECK: f1:
; CHECK-NEXT: # %bb.0:
; CHECK-NEXT: jmp     f0                          # TAILCALL

; CHECK: f2:
; CHECK-NEXT: # %bb.0:
; CHECK-NEXT: jmp     malloc                          # TAILCALL

define ptr @f1(i64 %count) "patchable-function"="prologue-short-redirect" {
entry:
  %call = tail call ptr @f0(i64 %count)
  ret ptr %call
}

declare ptr @f0(i64)

define noalias ptr @f2(i64 %count) "patchable-function"="prologue-short-redirect" {
entry:
  %call = tail call ptr @malloc(i64 %count)
  ret ptr %call
}

declare noalias ptr @malloc(i64) #0

attributes #0 = { allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{i32 1, !"MaxTLSAlign", i32 65536}
