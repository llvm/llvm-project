# Instructions that are valid for the current ISA but should be rejected by the assembler (e.g.
# invalid set of operands or operand's restrictions not met).

# RUN: not llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32r2 2>%t1
# RUN: FileCheck %s < %t1

        .text
        .set noreorder
        cache -1, 255($7)    # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
        cache 32, 255($7)    # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
        # FIXME: Check '0 < pos + size <= 32' constraint on ext
        ext $2, $3, -1, 1    # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        ext $2, $3, 32, 1    # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        ext $2, $3, 1, 0     # CHECK: :[[@LINE]]:24: error: expected immediate in range 1 .. 32
        ext $2, $3, 1, 33    # CHECK: :[[@LINE]]:24: error: expected immediate in range 1 .. 32
        # FIXME: Check size on ins
        ins $2, $3, -1, 1    # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        ins $2, $3, 32, 1    # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        jalr.hb $31          # CHECK: :[[@LINE]]:9: error: source and destination must be different
        jalr.hb $31, $31     # CHECK: :[[@LINE]]:9: error: source and destination must be different
        pref -1, 255($7)     # CHECK: :[[@LINE]]:14: error: expected 5-bit unsigned immediate
        pref 32, 255($7)     # CHECK: :[[@LINE]]:14: error: expected 5-bit unsigned immediate
        sll $2, $3, -1       # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        sll $2, $3, 32       # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        srl $2, $3, -1       # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        srl $2, $3, 32       # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        sra $2, $3, -1       # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        sra $2, $3, 32       # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        rotr $2, $3, -1      # CHECK: :[[@LINE]]:22: error: expected 5-bit unsigned immediate
        rotr $2, $3, 32      # CHECK: :[[@LINE]]:22: error: expected 5-bit unsigned immediate
