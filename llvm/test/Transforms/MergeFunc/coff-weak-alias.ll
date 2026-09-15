; RUN: split-file %s %t
; RUN: opt -S -passes=mergefunc %t/coff.ll | FileCheck %s --check-prefix=COFF
; RUN: opt -S -passes=mergefunc %t/coff-noalias.ll | \
; RUN:   FileCheck %s --check-prefix=NOALIAS
; RUN: opt -S -passes=mergefunc -mergefunc-use-aliases %t/coff-odr.ll | \
; RUN:   FileCheck %s --check-prefix=ODR
; RUN: opt -S -passes=mergefunc %t/elf.ll | FileCheck %s --check-prefix=ELF

; On COFF the merged body is named after its structural hash and put in a
; COMDAT, so every object defining the alias names the same fallback symbol.

;--- coff.ll
target triple = "x86_64-pc-windows-msvc"

; COFF: $[[BODY:"__llvm_mergefunc\$[0-9a-f]+"]] = comdat exactmatch
; COFF: @alias_g = weak alias i32 (i32), ptr @[[BODY]]
; COFF: define linkonce_odr {{.*}}i32 @[[BODY]](i32 %x) comdat
; COFF-NOT: define {{.*}} @g

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

; NOALIAS-NOT: comdat
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

; ODR: $[[BODY:"__llvm_mergefunc\$[0-9a-f]+"]] = comdat exactmatch
; ODR-DAG: @g = weak_odr unnamed_addr alias i32 (i32), ptr @[[BODY]]
; ODR-DAG: @f = weak_odr unnamed_addr alias i32 (i32), ptr @[[BODY]]
; ODR: define linkonce_odr i32 @[[BODY]](i32 %x) unnamed_addr comdat

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

; ELF-NOT: comdat
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
