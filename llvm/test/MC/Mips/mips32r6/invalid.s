# Instructions that are available for the current ISA but should be rejected by
# the assembler (e.g. invalid set of operands or operand's restrictions not met).

# RUN: not llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32r6 2>%t1
# RUN: FileCheck %s < %t1

        .text
local_label:
        .set noreorder
        .set noat
        align   $4, $2, $3, -1    # CHECK: :[[@LINE]]:29: error: expected 2-bit unsigned immediate
        align   $4, $2, $3, 4     # CHECK: :[[@LINE]]:29: error: expected 2-bit unsigned immediate
        jalr.hb $31 # CHECK: :[[@LINE]]:9: error: source and destination must be different
        jalr.hb $31, $31 # CHECK: :[[@LINE]]:9: error: source and destination must be different
        swc2    $25,24880($s0)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        break -1          # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
        break 1024        # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
        break -1, 5       # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
        break 1024, 5     # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
        break 7, -1       # CHECK: :[[@LINE]]:18: error: expected 10-bit unsigned immediate
        break 7, 1024     # CHECK: :[[@LINE]]:18: error: expected 10-bit unsigned immediate
        break 1024, 1024  # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
        lh  $33, 8($4)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lhe $34, 8($2)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lhu $35, 8($2)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lhue $36, 8($2)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lh  $2, 8($34)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
        lhe $4, 8($33)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        lhu $4, 8($35)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
        lhue $4, 8($37)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        lh  $2, -65536($4) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
        lh  $2, 65536($4)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
        lhe $4, -512($2)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        lhe $4, 512($2)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        lhu $4, -65536($2) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
        lhu $4, 65536($2)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
        lhue $4, -512($2)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        lhue $4, 512($2)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        // FIXME: Following tests are temporarily disabled, until "PredicateControl not in hierarchy" problem is resolved
        bltl  $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bltul $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        blel  $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bleul $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bgel  $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bgeul $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bgtl  $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bgtul $7, $8, local_label  # -CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bgec  $0, $2, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
        bltc  $0, $2, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
        bgeuc $0, $2, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
        bltuc $0, $2, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
        beqc  $0, $2, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
        bnec  $0, $2, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
        bgec  $2, $2, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: registers must be different
        bltc  $2, $2, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: registers must be different
        bgeuc $2, $2, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: registers must be different
        bltuc $2, $2, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: registers must be different
        beqc  $2, $2, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: registers must be different
        bnec  $2, $2, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: registers must be different
        blezc $0, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
        bgezc $0, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
        bgtzc $0, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
        bltzc $0, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
        beqzc $0, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
        bnezc $0, local_label # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
        cache -1, 255($7)    # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
        cache 32, 255($7)    # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
        jalr.hb $31          # CHECK: :[[@LINE]]:9: error: source and destination must be different
        jalr.hb $31, $31     # CHECK: :[[@LINE]]:9: error: source and destination must be different
        ldc2 $20, -1025($s2) # CHECK: :[[@LINE]]:9: error: instruction requires a CPU feature not currently enabled
        ldc2 $20, 1024($s2)  # CHECK: :[[@LINE]]:9: error: instruction requires a CPU feature not currently enabled
        lsa $2, $3, $4, 0    # CHECK: :[[@LINE]]:25: error: expected immediate in range 1 .. 4
        lsa $2, $3, $4, 5    # CHECK: :[[@LINE]]:25: error: expected immediate in range 1 .. 4
        pref -1, 255($7)     # CHECK: :[[@LINE]]:14: error: expected 5-bit unsigned immediate
        pref 32, 255($7)     # CHECK: :[[@LINE]]:14: error: expected 5-bit unsigned immediate
        mtc0  $4, $3, -1     # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
        mtc0  $4, $3, 8      # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
        mtc2  $4, $3, -1     # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
        mtc2  $4, $3, 8      # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
        mfc0  $4, $3, -1     # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
        mfc0  $4, $3, 8      # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
        mfc2  $4, $3, -1     # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
        mfc2  $4, $3, 8      # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
        sdc2 $20, -1025($s2) # CHECK: :[[@LINE]]:9: error: instruction requires a CPU feature not currently enabled
        sdc2 $20, 1024($s2)  # CHECK: :[[@LINE]]:9: error: instruction requires a CPU feature not currently enabled
        sync -1              # CHECK: :[[@LINE]]:14: error: expected 5-bit unsigned immediate
        sync 32              # CHECK: :[[@LINE]]:14: error: expected 5-bit unsigned immediate
