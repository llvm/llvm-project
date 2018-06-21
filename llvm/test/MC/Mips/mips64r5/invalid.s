# Instructions that are valid for the current ISA but should be rejected by the assembler (e.g.
# invalid set of operands or operand's restrictions not met).

# RUN: not llvm-mc %s -triple=mips64-unknown-linux -mcpu=mips64r5 2>%t1
# RUN: FileCheck %s < %t1

        .text
        .set noreorder
        cache -1, 255($7)    # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
        cache 32, 255($7)    # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
        drotr32 $2, $3, -1   # CHECK: :[[@LINE]]:25: error: expected 5-bit unsigned immediate
        drotr32 $2, $3, 32   # CHECK: :[[@LINE]]:25: error: expected 5-bit unsigned immediate
        jalr.hb $31          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: source and destination must be different
        jalr.hb $31, $31     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: source and destination must be different
        pref -1, 255($7)     # CHECK: :[[@LINE]]:14: error: expected 5-bit unsigned immediate
        pref 32, 255($7)     # CHECK: :[[@LINE]]:14: error: expected 5-bit unsigned immediate
        dmtc0  $4, $3, -1    # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
        dmtc0  $4, $3, 8     # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
        dmfc0  $4, $3, -1    # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
        dmfc0  $4, $3, 8     # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
        sd $32, 65536($32)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lb $32, 8($5)        # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
        lb $4, -32769($5)    # CHECK: :[[@LINE]]:16: error: expected memory with 16-bit signed offset
        lb $4, 32768($5)     # CHECK: :[[@LINE]]:16: error: expected memory with 16-bit signed offset
        lb $4, 8($32)        # CHECK: :[[@LINE]]:16: error: expected memory with 16-bit signed offset
        lbu $32, 8($5)       # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
        lbu $4, -32769($5)   # CHECK: :[[@LINE]]:17: error: expected memory with 16-bit signed offset
        lbu $4, 32768($5)    # CHECK: :[[@LINE]]:17: error: expected memory with 16-bit signed offset
        lbu $4, 8($32)       # CHECK: :[[@LINE]]:17: error: expected memory with 16-bit signed offset
        ldc1 $f32, 300($10)   # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
        ldc1 $f7, -32769($10) # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        ldc1 $f7, 32768($10)  # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        ldc1 $f7, 300($32)    # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        sdc1 $f32, 64($10)    # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
        sdc1 $f7, -32769($10) # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        sdc1 $f7, 32768($10)  # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        sdc1 $f7, 64($32)     # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        lwc1 $f32, 32($5)     # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
        lwc1 $f2, -32769($5)  # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        lwc1 $f2, 32768($5)   # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        lwc1 $f2, 32($32)     # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        swc1 $f32, 369($13)   # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
        swc1 $f6, -32769($13) # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        swc1 $f6, 32768($13)  # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        swc1 $f6, 369($32)    # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        ldc2 $32, 1023($12)  # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
        ldc2 $1, -32769($12) # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        ldc2 $1, 32768($12)  # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        ldc2 $1, 1023($32)   # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        sdc2 $32, 8($16)     # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
        sdc2 $1, -32769($16) # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        sdc2 $1, 32768($16)  # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        sdc2 $1, 8($32)      # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        lwc2 $32, 16($4)     # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
        lwc2 $1, -32769($4)  # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        lwc2 $1, 32768($4)   # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        lwc2 $1, 16($32)     # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        swc2 $32, 777($17)   # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
        swc2 $1, -32769($17) # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        swc2 $1, 32768($17)  # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        swc2 $1, 777($32)    # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        lwc2 $11, -32769($12)   # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        lwc2 $11, 32768($12)    # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        sdc2 $11, -32769($12)   # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        sdc2 $11, 32768($12)    # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        swc2 $11, -32769($12)   # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
        swc2 $11, 32768($12)    # CHECK: :[[@LINE]]:19: error: expected memory with 16-bit signed offset
