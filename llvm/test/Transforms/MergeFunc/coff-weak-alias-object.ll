; The weak external must name a symbol every object defining the alias agrees
; on. Any COFF target will do; this is a property of the object format.

; REQUIRES: aarch64-registered-target
; RUN: opt -S -passes=mergefunc %s | \
; RUN:   llc -filetype=obj -mtriple=aarch64-pc-windows-msvc -o %t.obj
; RUN: llvm-readobj --symbols %t.obj | FileCheck %s

; CHECK:      Name: alias_g
; CHECK:      StorageClass: WeakExternal
; CHECK:      AuxWeakExternal {
; CHECK-NEXT:   Linked: __llvm_mergefunc$
; CHECK-NOT:  .weak.alias_g.default

target triple = "aarch64-pc-windows-msvc"

@alias_g = weak alias i32 (i32), ptr @g

define internal i32 @f(i32 %x) {
  %a = mul i32 %x, 3
  %b = add i32 %a, 7
  ret i32 %b
}

define linkonce_odr i32 @g(i32 %x) unnamed_addr {
  %a = mul i32 %x, 3
  %b = add i32 %a, 7
  ret i32 %b
}

; A local body would instead get .weak.alias_g.default.keep, named after this.
define i32 @keep() {
  %r = call i32 @alias_g(i32 1)
  ret i32 %r
}
