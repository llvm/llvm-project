; RUN: split-file %s %t
; RUN: opt -S -passes=mergefunc %t/coff.ll | FileCheck %s --check-prefix=COFF
; RUN: opt -S -passes=mergefunc %t/coff-noalias.ll | \
; RUN:   FileCheck %s --check-prefix=NOALIAS
; RUN: opt -S -passes=mergefunc -mergefunc-use-aliases %t/coff-odr.ll | \
; RUN:   FileCheck %s --check-prefix=ODR
; RUN: opt -S -passes=mergefunc %t/elf.ll | FileCheck %s --check-prefix=ELF

; A COFF weak external names the symbol it resolves to, and a local symbol
; cannot be named, so an alias must keep pointing at an external definition.

;--- coff.ll
target triple = "x86_64-pc-windows-msvc"

; @f is local, so @g is kept as a thunk rather than replaced, and the alias
; goes on naming @g.
; COFF: @alias_g = weak alias i32 (i32), ptr @g
; COFF: define internal i32 @f
; COFF: define linkonce_odr i32 @g

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

;--- coff-noalias.ll
target triple = "x86_64-pc-windows-msvc"

; Without an alias naming it, @g is replaced as usual.
; NOALIAS: define internal i32 @f
; NOALIAS: call i32 @f

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

define i32 @use() {
  %r = call i32 @g(i32 1)
  ret i32 %r
}

;--- coff-odr.ll
target triple = "x86_64-pc-windows-msvc"

; The shared body is private, so both halves get a thunk instead of an alias
; even though aliases are enabled.
; ODR-NOT: alias i32
; ODR-DAG: define weak_odr i32 @f
; ODR-DAG: define weak_odr i32 @g

define weak_odr i32 @f(i32 %x) unnamed_addr {
  %a = mul i32 %x, 3
  %b = add i32 %a, 7
  %c = add i32 %b, 1
  ret i32 %c
}

define weak_odr i32 @g(i32 %x) unnamed_addr {
  %a = mul i32 %x, 3
  %b = add i32 %a, 7
  %c = add i32 %b, 1
  ret i32 %c
}

;--- elf.ll
target triple = "x86_64-unknown-linux-gnu"

; Other formats can alias a local symbol, so @g is replaced there.
; ELF: @alias_g = weak alias i32 (i32), ptr @f
; ELF: define internal i32 @f
; ELF-NOT: define {{.*}} @g

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
