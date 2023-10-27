; RUN: llc -mtriple=arm64-apple-none-elf %s -o - | FileCheck %s --check-prefix CHECK-ELF
; RUN: llc -mtriple=arm64-apple-none-macho %s -o - | FileCheck %s --check-prefix CHECK-MACHO

@var = global i8 0

define i8 @foo() {
  %x = load i8, ptr @var

  ; CHECK-ELF: adrp x{{[0-9]+}}, :got:var
  ; CHECK-ELF: ldr x{{[0-9]+}}, [x{{[0-9]+}}, :got_lo12:var]

  ; CHECK-MACHO: adrp x{{[0-9]+}}, _var@PAGE
  ; CHECK-MACHO: ldrb w{{[0-9]+}}, [x{{[0-9]+}}, _var@PAGEOFF]

  ret i8 %x
}
