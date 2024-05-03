// RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r - | FileCheck %s

        .section .data
this:
        .word this@GOTPCREL
        .word extern_sym@GOTPCREL+4
        .word negative_offset@GOTPCREL-4

// CHECK:      Section ({{.*}}) .rela.data
// CHECK-NEXT:   0x0 R_RISCV_GOT32_PCREL this 0x0
// CHECK-NEXT:   0x4 R_RISCV_GOT32_PCREL extern_sym 0x4
// CHECK-NEXT:   0x8 R_RISCV_GOT32_PCREL negative_offset 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT: }
