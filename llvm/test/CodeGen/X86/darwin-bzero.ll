; RUN: llc < %s -mtriple=i386-apple-darwin10 | FileCheck -check-prefixes=CHECK,BZERO %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 | FileCheck -check-prefixes=CHECK,BZERO %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck -check-prefixes=CHECK,NOBZERO %s
; RUN: llc < %s -mtriple=x86_64-apple-ios10.0-simulator | FileCheck -check-prefixes=CHECK,NOBZERO %s

declare void @llvm.memset.p0.i32(ptr nocapture, i8, i32, i1) nounwind

; CHECK-LABEL: foo:
; BZERO: {{calll|callq}} ___bzero
; NOBZERO-NOT: bzero
define void @foo(ptr %p, i32 %len) {
  call void @llvm.memset.p0.i32(ptr %p, i8 0, i32 %len, i1 false)
  ret void
}
