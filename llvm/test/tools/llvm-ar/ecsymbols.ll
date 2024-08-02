;; Test that ECSYMBOLS section is created when ARM64EC bitcode is used.

; RUN: llvm-as %s -o %t.bc
; RUN: rm -f %t.a
; RUN: llvm-ar cr %t.a %t.bc
; RUN: llvm-nm --print-armap %t.a | FileCheck %s

; CHECK-NOT: Archive map
; CHECK: Archive EC map
; CHECK-NEXT: a in ecsymbols.ll.tmp.bc
; CHECK-NEXT: b in ecsymbols.ll.tmp.bc
; CHECK-NEXT: c in ecsymbols.ll.tmp.bc
; CHECK-EMPTY:

target triple = "arm64ec-unknown-windows-msvc"

define void @b() { ret void }
define void @c() { ret void }
define void @a() { ret void }
