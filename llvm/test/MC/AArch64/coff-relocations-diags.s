// RUN: not llvm-mc -triple aarch64-win32 -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

  adrp x0, :got:symbol
  // CHECK: error: relocation variant :got: unsupported on COFF targets
  // CHECK-NEXT: adrp x0, :got:symbol
  // CHECK-NEXT: ^

  ldr x0, [x0, :got_lo12:symbol]
  // CHECK: error: relocation variant :got_lo12: unsupported on COFF targets
  // CHECK-NEXT: ldr x0, [x0, :got_lo12:symbol]
  // CHECK-NEXT: ^

  adrp x0, :tlsdesc:symbol
  // CHECK: error: relocation variant :tlsdesc: unsupported on COFF targets
  // CHECK-NEXT: adrp x0, :tlsdesc:symbol
  // CHECK-NEXT: ^
  add x0, x0, :tlsdesc_lo12:symbol
  // CHECK: error: relocation variant :tlsdesc_lo12: unsupported on COFF targets
  // CHECK-NEXT: add x0, x0, :tlsdesc_lo12:symbol
  // CHECK-NEXT: ^

  adrp x0, :gottprel:symbol
  // CHECK: error: relocation variant :gottprel: unsupported on COFF targets
  // CHECK-NEXT: adrp x0, :gottprel:symbol
  // CHECK-NEXT: ^
  ldr x0, [x0, :gottprel_lo12:symbol]
  // CHECK: error: relocation variant :gottprel_lo12: unsupported on COFF targets
  // CHECK-NEXT: ldr x0, [x0, :gottprel_lo12:symbol]
  // CHECK-NEXT: ^

  add x0, x0, #:dtprel_hi12:symbol, lsl #12
  // CHECK: error: relocation variant :dtprel_hi12: unsupported on COFF targets
  // CHECK-NEXT: add x0, x0, #:dtprel_hi12:symbol, lsl #12
  // CHECK-NEXT: ^
  add x0, x0, :dtprel_lo12:symbol
  // CHECK: error: relocation variant :dtprel_lo12: unsupported on COFF targets
  // CHECK-NEXT: add x0, x0, :dtprel_lo12:symbol
  // CHECK-NEXT: ^

label:
  movz x0, #:abs_g0:symbol
  // CHECK: error: relocation type :abs_g0: unsupported on COFF targets
  // CHECK-NEXT: movz x0, #:abs_g0:symbol
  // CHECK-NEXT: ^

  .section .rdata, "dr"
table:
  .short label - table
  // CHECK: error: Cannot represent this expression
  // CHECK-NEXT: .short label - table
  // CHECK-NEXT: ^
