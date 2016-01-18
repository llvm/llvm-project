# Instructions that are valid for the current ISA but should be rejected by the assembler (e.g.
# invalid set of operands or operand's restrictions not met).

# RUN: not llvm-mc %s -triple=mips64-unknown-linux -mcpu=mips64r3 2>%t1
# RUN: FileCheck %s < %t1 -check-prefix=ASM

        .text
        .set noreorder
        jalr.hb $31 # ASM: :[[@LINE]]:9: error: source and destination must be different
        jalr.hb $31, $31 # ASM: :[[@LINE]]:9: error: source and destination must be different
