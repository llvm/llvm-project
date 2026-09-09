; RUN: split-file %s %t
; RUN: opt -S -passes=mergefunc %t/coff.ll | FileCheck %s --check-prefix=COFF
; RUN: opt -S -passes=mergefunc -mergefunc-use-aliases %t/coff.ll | FileCheck %s --check-prefix=COFF
; RUN: opt -S -passes=mergefunc %t/elf.ll | FileCheck %s --check-prefix=ELF

; On COFF an alias that other objects can see is emitted as a weak external,
; which names the symbol to fall back to, so it has to keep pointing at one that
; has a name. Everywhere else the alias can point straight at the merged body.

;--- coff.ll
target triple = "x86_64-pc-windows-msvc"

; COFF: @alias_g = weak alias i32 (i32), ptr @g
; COFF: define linkonce_odr i32 @g
; COFF-NEXT: %[[R:.*]] = tail call i32 @f
; COFF-NEXT: ret i32 %[[R]]

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

;--- elf.ll
target triple = "x86_64-unknown-linux-gnu"

; ELF: @alias_g = weak alias i32 (i32), ptr @f
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
