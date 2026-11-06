; A COFF weak external has to name the symbol it resolves to, so an alias must
; not be left pointing at a local symbol. Any COFF target will do; this is a
; property of the object format.

; REQUIRES: aarch64-registered-target
; RUN: opt -S -passes=mergefunc %s | \
; RUN:   llc -filetype=obj -mtriple=aarch64-pc-windows-msvc -o %t.obj
; RUN: llvm-readobj --symbols %t.obj | FileCheck %s

; CHECK:      Name: alias_g
; CHECK:      StorageClass: WeakExternal
; CHECK:      AuxWeakExternal {
; CHECK-NEXT:   Linked: g
; CHECK-NOT:  .weak.alias_g.default

target triple = "aarch64-pc-windows-msvc"

$g = comdat any

@alias_g = weak alias i32 (i32), ptr @g

define internal i32 @f(i32 %x) {
  %a = mul i32 %x, 3
  %b = add i32 %a, 7
  ret i32 %b
}

define linkonce_odr i32 @g(i32 %x) unnamed_addr comdat {
  %a = mul i32 %x, 3
  %b = add i32 %a, 7
  ret i32 %b
}

; Replacing @g would leave the alias naming a local symbol, for which the
; writer would invent .weak.alias_g.default.keep, named after this.
define i32 @keep() {
  %r = call i32 @alias_g(i32 1)
  ret i32 %r
}
