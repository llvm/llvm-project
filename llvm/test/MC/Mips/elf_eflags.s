// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux %s -o -| llvm-readobj -h | FileCheck %s

        .mips_hack_elf_flags 0x50001005

// CHECK: Flags [ (0x50001005)
