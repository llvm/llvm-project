# Instructions that are available for the current ISA but should be rejected by
# the assembler (e.g. invalid set of operands or operand's restrictions not met).

# RUN: not llvm-mc %s -triple=mips64-unknown-linux -mcpu=mips64r6 2>%t1
# RUN: FileCheck %s < %t1

        .text
local_label:
        .set noreorder
	.set noat
        align   $4, $2, $3, -1    # CHECK: :[[@LINE]]:29: error: expected 2-bit unsigned immediate
        align   $4, $2, $3, 4     # CHECK: :[[@LINE]]:29: error: expected 2-bit unsigned immediate
        jalr.hb $31 # CHECK: :[[@LINE]]:9: error: source and destination must be different
        jalr.hb $31, $31 # CHECK: :[[@LINE]]:9: error: source and destination must be different
        ldc2    $8,-21181($at)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        break 1024        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        break 1024, 5     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        break 7, 1024     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        break 1024, 1024  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        // FIXME: Following tests are temporarely disabled, until "PredicateControl not in hierarchy" problem is resolved
        bltl  $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bltul $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        blel  $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bleul $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bgel  $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bgeul $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bgtl  $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bgtul $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dalign  $4, $2, $3, -1    # CHECK: :[[@LINE]]:29: error: expected 3-bit unsigned immediate
        dalign  $4, $2, $3, 8     # CHECK: :[[@LINE]]:29: error: expected 3-bit unsigned immediate
        dlsa    $2, $3, $4, 0     # CHECK: :[[@LINE]]:29: error: expected immediate in range 1 .. 4
        dlsa    $2, $3, $4, 5     # CHECK: :[[@LINE]]:29: error: expected immediate in range 1 .. 4
        lsa     $2, $3, $4, 0     # CHECK: :[[@LINE]]:29: error: expected immediate in range 1 .. 4
        lsa     $2, $3, $4, 5     # CHECK: :[[@LINE]]:29: error: expected immediate in range 1 .. 4
