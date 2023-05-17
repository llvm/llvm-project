// RUN: llvm-mc -triple aarch64-elf -filetype=obj %s -o - | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s

// CHECK: adr x0, 0x64
// CHECK-NEXT: adr x2, 0x4
// CHECK-NEXT: R_AARCH64_ADR_PREL_LO21	Symbol
// CHECK-NEXT: adr x3, 0x8
// CHECK-NEXT: R_AARCH64_ADR_PREL_LO21	Symbol
// CHECK-NEXT: adr x4, 0xc
// CHECK-NEXT: R_AARCH64_ADR_PREL_LO21	Symbol+0xf1000
// CHECK-NEXT: adr x5, 0x10
// CHECK-NEXT: R_AARCH64_ADR_PREL_LO21	Symbol+0xf1000
// CHECK-NEXT: adr x6, 0x14
// CHECK-NEXT: R_AARCH64_ADR_PREL_LO21	Symbol+0xf1000

  adr x0, 100
  adr x2, Symbol
  adr x3, Symbol + 0
  adr x4, Symbol + 987136
  adr x5, (0xffffffff000f1000 - 0xffffffff00000000 + Symbol)
  adr x6, Symbol + (0xffffffff000f1000 - 0xffffffff00000000)

// CHECK-NEXT: adrp x0, 0x0
// CHECK-NEXT: R_AARCH64_ADR_PREL_PG_HI21	Symbol
// CHECK-NEXT: adrp x2, 0x0
// CHECK-NEXT: R_AARCH64_ADR_PREL_PG_HI21	Symbol
// CHECK-NEXT: adrp x3, 0x0
// CHECK-NEXT: R_AARCH64_ADR_PREL_PG_HI21	Symbol+0xf1000
// CHECK-NEXT: adrp x4, 0x0
// CHECK-NEXT: R_AARCH64_ADR_PREL_PG_HI21	Symbol+0xf1000
// CHECK-NEXT: adrp x5, 0x0
// CHECK-NEXT: R_AARCH64_ADR_PREL_PG_HI21	Symbol+0xf1000

  adrp x0, Symbol
  adrp x2, Symbol + 0
  adrp x3, Symbol + 987136
  adrp x4, (0xffffffff000f1000 - 0xffffffff00000000 + Symbol)
  adrp x5, Symbol + (0xffffffff000f1000 - 0xffffffff00000000)
