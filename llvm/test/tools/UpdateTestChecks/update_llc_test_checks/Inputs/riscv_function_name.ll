; Check that we accept functions with '$' in the name.

; RUN: llc -mtriple=riscv32-unknown-linux < %s | FileCheck %s
; RUN: llc -mtriple=riscv32-apple-none-macho < %s | FileCheck %s --check-prefix=MACHO

define hidden i32 @"_Z54bar$ompvariant$bar"() {
entry:
  ret i32 2
}
