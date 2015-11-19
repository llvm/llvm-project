; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core2 | FileCheck %s --check-prefix=CORE2
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=nehalem | FileCheck %s --check-prefix=NEHALEM
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=btver2 | FileCheck %s --check-prefix=BTVER2

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1)

define void @copy16bytes(i8* nocapture %a, i8* nocapture readonly %b) {
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 16, i32 1, i1 false)
  ret void

  ; CHECK-LABEL: copy16bytes
  ; CORE2: movq
  ; CORE2-NEXT: movq
  ; CORE2-NEXT: movq
  ; CORE2-NEXT: movq
  ; CORE2-NEXT: retq

  ; NEHALEM: movups
  ; NEHALEM-NEXT: movups
  ; NEHALEM-NEXT: retq

  ; BTVER2: movups
  ; BTVER2-NEXT: movups
  ; BTVER2-NEXT: retq
}
