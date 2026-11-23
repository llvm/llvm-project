// RUN: llvm-mc -triple aarch64-windows -filetype obj -o %t.obj %s
// RUN: llvm-mc -triple arm64ec-windows -filetype obj -o %t-ec.obj %s
// RUN: llvm-readobj -r --syms %t.obj | FileCheck %s
// RUN: llvm-readobj -r --syms %t-ec.obj | FileCheck %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.obj | FileCheck %s --check-prefix=DISASM
// RUN: llvm-objdump --no-print-imm-hex -d %t-ec.obj | FileCheck %s --check-prefix=DISASM

  .bss
  .zero 0xff0
.Lsmall:
  .long 0
  .zero 28
.Llarge:
  .long 0
large:
  .long 0

  .text
f:
  add x0, x0, :secrel_hi12:.Lsmall
  add x0, x0, :secrel_lo12:.Lsmall
  add x1, x1, :secrel_hi12:.Lsmall+16
  add x1, x1, :secrel_lo12:.Lsmall+16
  add x2, x2, :secrel_hi12:.Llarge
  add x2, x2, :secrel_lo12:.Llarge
  ldr w3, [x3, :secrel_lo12:.Llarge+4]
  add x4, x4, :secrel_hi12:large
  add x4, x4, :secrel_lo12:large
  add x5, x5, :secrel_hi12:.Lsmall+4
  add x5, x5, :secrel_lo12:.Lsmall+4
  add x6, x6, :secrel_hi12:large+128
  add x6, x6, :secrel_lo12:large+128
  ret

// CHECK:      Relocations [
// CHECK:        Section ({{[0-9]+}}) .text {
// CHECK-NEXT:     0x0 IMAGE_REL_ARM64_SECREL_HIGH12A .bss (
// CHECK-NEXT:     0x4 IMAGE_REL_ARM64_SECREL_LOW12A .bss (
// CHECK-NEXT:     0x8 IMAGE_REL_ARM64_SECREL_HIGH12A $L.bss_secrel_1 (
// CHECK-NEXT:     0xC IMAGE_REL_ARM64_SECREL_LOW12A $L.bss_secrel_1 (
// CHECK-NEXT:     0x10 IMAGE_REL_ARM64_SECREL_HIGH12A .Llarge (
// CHECK-NEXT:     0x14 IMAGE_REL_ARM64_SECREL_LOW12A .Llarge (
// CHECK-NEXT:     0x18 IMAGE_REL_ARM64_SECREL_LOW12L $L.bss_secrel_2 (
// CHECK-NEXT:     0x1C IMAGE_REL_ARM64_SECREL_HIGH12A large (
// CHECK-NEXT:     0x20 IMAGE_REL_ARM64_SECREL_LOW12A large (
// CHECK-NEXT:     0x24 IMAGE_REL_ARM64_SECREL_HIGH12A .bss (
// CHECK-NEXT:     0x28 IMAGE_REL_ARM64_SECREL_LOW12A .bss (
// CHECK-NEXT:     0x2C IMAGE_REL_ARM64_SECREL_HIGH12A large (
// CHECK-NEXT:     0x30 IMAGE_REL_ARM64_SECREL_LOW12A large (
// CHECK-NEXT:   }

// CHECK:      Symbol {
// CHECK:        Name: large
// CHECK-NEXT:   Value: 4116
// CHECK-NEXT:   Section: .bss
// CHECK:        StorageClass: Static

// CHECK:      Symbol {
// CHECK:        Name: $L.bss_secrel_1
// CHECK-NEXT:   Value: 4096
// CHECK-NEXT:   Section: .bss
// CHECK:        StorageClass: Label

// CHECK:      Symbol {
// CHECK:        Name: .Llarge
// CHECK-NEXT:   Value: 4112
// CHECK-NEXT:   Section: .bss
// CHECK:        StorageClass: Label

// CHECK:      Symbol {
// CHECK:        Name: $L.bss_secrel_2
// CHECK-NEXT:   Value: 4116
// CHECK-NEXT:   Section: .bss
// CHECK:        StorageClass: Label

// DISASM-LABEL: <f>:
// DISASM-NEXT:    add x0, x0, #4080, lsl #12
// DISASM-NEXT:    add x0, x0, #4080
// DISASM-NEXT:    add x1, x1, #0, lsl #12
// DISASM-NEXT:    add x1, x1, #0
// DISASM-NEXT:    add x2, x2, #0, lsl #12
// DISASM-NEXT:    add x2, x2, #0
// DISASM-NEXT:    ldr w3, [x3]
// DISASM-NEXT:    add x4, x4, #0, lsl #12
// DISASM-NEXT:    add x4, x4, #0
// DISASM-NEXT:    add x5, x5, #4084, lsl #12
// DISASM-NEXT:    add x5, x5, #4084
// DISASM-NEXT:    add x6, x6, #128, lsl #12
// DISASM-NEXT:    add x6, x6, #128
// DISASM-NEXT:    ret
