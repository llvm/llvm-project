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
        aui     $4, $4, 65536     # CHECK: :[[@LINE]]:25: error: expected 16-bit unsigned immediate
        aui     $4, $4, -1        # CHECK: :[[@LINE]]:25: error: expected 16-bit unsigned immediate
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
        lh  $33, 8($4)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
        lhe $34, 8($2)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
        lhu $35, 8($2)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
        lhue $36, 8($2)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
        lh  $2, 8($34)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
        lhe $4, 8($33)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
        lhu $4, 8($35)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
        lhue $4, 8($37)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
        lh  $2, -65536($4) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
        lh  $2, 65536($4)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
        lhe $4, -512($2)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lhe $4, 512($2)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lhu $4, -65536($2) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
        lhu $4, 65536($2)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
        lhue $4, -512($2)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lhue $4, 512($2)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
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
        bgec  $2, $4, -131076    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bgec  $2, $4, -131071    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bgec  $2, $4, 131072     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bgec  $2, $4, 131071     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bltc  $2, $4, -131076    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bltc  $2, $4, -131071    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bltc  $2, $4, 131072     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bltc  $2, $4, 131071     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bgeuc  $2, $4, -131076   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bgeuc  $2, $4, -131071   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bgeuc  $2, $4, 131072    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bgeuc  $2, $4, 131071    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bltuc  $2, $4, -131076   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bltuc  $2, $4, -131071   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bltuc  $2, $4, 131072    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bltuc  $2, $4, 131071    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        beqc  $2, $4, -131076    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        beqc  $2, $4, -131071    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        beqc  $2, $4, 131072     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        beqc  $2, $4, 131071     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bnec  $2, $4, -131076    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bnec  $2, $4, -131071    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bnec  $2, $4, 131072     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bnec  $2, $4, 131071     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        blezc $2, -131076        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        blezc $2, -131071        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        blezc $2, 131072         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        blezc $2, 131071         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bgezc $2, -131076        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bgezc $2, -131071        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bgezc $2, 131072         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bgezc $2, 131071         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bgtzc $2, -131076        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bgtzc $2, -131071        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bgtzc $2, 131072         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bgtzc $2, 131071         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bltzc $2, -131076        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bltzc $2, -131071        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bltzc $2, 131072         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bltzc $2, 131071         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        beqzc $2, -4194308       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        beqzc $2, -4194303       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        beqzc $2, 4194304        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        beqzc $2, 4194303        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bnezc $2, -4194308       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bnezc $2, -4194303       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        bnezc $2, 4194304        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
        bnezc $2, 4194303        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
        cache -1, 255($7)    # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
        cache 32, 255($7)    # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
        dvp $17, $3          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        dvp $17, 3           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        dvp 3                # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        evp $16, $3          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        evp $16, 3           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        evp 3                # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        jalr.hb $31          # CHECK: :[[@LINE]]:9: error: source and destination must be different
        jalr.hb $31, $31     # CHECK: :[[@LINE]]:9: error: source and destination must be different
        lapc $7, 1048576     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 19-bit signed immediate and multiple of 4
        lapc $6, -1048580    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 19-bit signed immediate and multiple of 4
        lapc $3, 3           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 19-bit signed immediate and multiple of 4
        lapc $3, -1          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 19-bit signed immediate and multiple of 4
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
        lb $32, 8($5)        # CHECK: :[[@LINE]]:12: error: invalid register number
        lb $4, -2147483649($5)  # CHECK: :[[@LINE]]:16: error: expected memory with 32-bit signed offset
        lb $4, 2147483648($5)   # CHECK: :[[@LINE]]:16: error: expected memory with 32-bit signed offset
        lb $4, 8($32)        # CHECK: :[[@LINE]]:18: error: invalid register number
        lbu $32, 8($5)       # CHECK: :[[@LINE]]:13: error: invalid register number
        lbu $4, -2147483649($5) # CHECK: :[[@LINE]]:17: error: expected memory with 32-bit signed offset
        lbu $4, 2147483648($5)  # CHECK: :[[@LINE]]:17: error: expected memory with 32-bit signed offset
        lbu $4, 8($32)       # CHECK: :[[@LINE]]:19: error: invalid register number
        ldc1 $f32, 300($10)   # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
        ldc1 $f7, -32769($10) # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        ldc1 $f7, 32768($10)  # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        ldc1 $f7, 300($32)    # CHECK: :[[@LINE]]:23: error: invalid register number
        sdc1 $f32, 64($10)    # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
        sdc1 $f7, -32769($10) # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        sdc1 $f7, 32768($10)  # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        sdc1 $f7, 64($32)     # CHECK: :[[@LINE]]:22: error: invalid register number
        lwc1 $f32, 32($5)     # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
        lwc1 $f2, -32769($5)  # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        lwc1 $f2, 32768($5)   # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        lwc1 $f2, 32($32)     # CHECK: :[[@LINE]]:22: error: invalid register number
        swc1 $f32, 369($13)   # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
        swc1 $f6, -32769($13) # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        swc1 $f6, 32768($13)  # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        swc1 $f6, 369($32)    # CHECK: :[[@LINE]]:23: error: invalid register number
        ldc2 $32, 1023($12)  # CHECK: :[[@LINE]]:14: error: invalid register number
        ldc2 $11, -1025($12) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ldc2 $11, 1024($12)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sdc2 $32, 8($16)     # CHECK: :[[@LINE]]:14: error: invalid register number
        sdc2 $11, -1025($12) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sdc2 $11, 1024($12)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        lwc2 $32, 16($4)     # CHECK: :[[@LINE]]:14: error: invalid register number
        lwc2 $11, -1025($12) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        lwc2 $11, 1024($12)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        swc2 $32, 777($17)   # CHECK: :[[@LINE]]:14: error: invalid register number
        swc2 $11, -1025($12) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        swc2 $11, 1024($12)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
