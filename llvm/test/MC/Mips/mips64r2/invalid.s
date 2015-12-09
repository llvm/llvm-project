# Instructions that are valid for the current ISA but should be rejected by the assembler (e.g.
# invalid set of operands or operand's restrictions not met).

# RUN: not llvm-mc %s -triple=mips64-unknown-linux -mcpu=mips64r2 2>%t1
# RUN: FileCheck %s < %t1

        .text
        .set noreorder
        cache -1, 255($7)    # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
        cache 32, 255($7)    # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
        # FIXME: Check size on dext*
        dext $2, $3, -1, 1   # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
        dext $2, $3, 64, 1   # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
        dextm $2, $3, -1, 1  # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
        dextm $2, $3, 32, 1  # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
        dextu $2, $3, 31, 1  # CHECK: :[[@LINE]]:23: error: expected immediate in range 32 .. 63
        dextu $2, $3, 64, 1  # CHECK: :[[@LINE]]:23: error: expected immediate in range 32 .. 63
        # FIXME: Check size on dins*
        dins $2, $3, -1, 1   # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
        dins $2, $3, 64, 1   # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
        dinsm $2, $3, -1, 1  # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
        dinsm $2, $3, 32, 1  # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
        dinsu $2, $3, 31, 1  # CHECK: :[[@LINE]]:23: error: expected immediate in range 32 .. 63
        dinsu $2, $3, 64, 1  # CHECK: :[[@LINE]]:23: error: expected immediate in range 32 .. 63
        drotr $2, $3, -1     # CHECK: :[[@LINE]]:23: error: expected 6-bit unsigned immediate
        drotr $2, $3, 64     # CHECK: :[[@LINE]]:23: error: expected 6-bit unsigned immediate
        drotr32 $2, $3, -1   # CHECK: :[[@LINE]]:25: error: expected 5-bit unsigned immediate
        drotr32 $2, $3, 32   # CHECK: :[[@LINE]]:25: error: expected 5-bit unsigned immediate
        dsll $2, $3, -1      # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
        dsll $2, $3, 64      # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
        dsll32 $2, $3, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
        dsll32 $2, $3, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
        dsrl $2, $3, -1      # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
        dsrl $2, $3, 64      # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
        dsrl32 $2, $3, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
        dsrl32 $2, $3, 64    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
        dsra $2, $3, -1      # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
        dsra $2, $3, 64      # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
        dsra32 $2, $3, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
        dsra32 $2, $3, 64    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
        # FIXME: Check size on ext
        ext $2, $3, -1, 1    # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        ext $2, $3, 32, 1    # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        # FIXME: Check size on ins
        ins $2, $3, -1, 1    # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        ins $2, $3, 32, 1    # CHECK: :[[@LINE]]:21: error: expected 5-bit unsigned immediate
        jalr.hb $31          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: source and destination must be different
        jalr.hb $31, $31     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: source and destination must be different
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
