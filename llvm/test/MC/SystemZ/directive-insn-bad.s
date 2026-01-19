# RUN: not llvm-mc -triple s390x-linux-gnu %s 2>&1 | FileCheck %s

# CHECK: error: unrecognized format
# CHECK: .insn not_a_format,0x0
        .insn not_a_format,0x0

# CHECK: error: unexpected token in directive
# CHECK: .insn rr,0x0101
        .insn rr,0x0101

# CHECK: error: unexpected token at start of statement
# CHECK: .insn e,0x0101,0
        .insn e,0x0101,0

# CHECK: error: unexpected token in directive
# CHECK: .insn rr,0x1800,0
        .insn rr,0x1800,0

# CHECK: error: unexpected token at start of statement
# CHECK: .insn rr,0x1800,%r1,0(%r2)
        .insn rr,0x1800,%r1,0(%r2)

# CHECK: error: unknown token in expression
# CHECK: .insn rxy_a,0xe30000000016,%r1,%r2,0
        .insn rxy_a,0xe30000000016,%r1,%r2,0

# CHECK: error: unknown token in expression
# CHECK: .insn ril_c,0xc00400000000,%r1,0
        .insn ril_c,0xc00400000000,%r1,0

# CHECK: error: invalid operand for instruction
# CHECK: .insn vri_e,0xe7000000004a,%r1,%v2,837,7,6
        .insn vri_e,0xe7000000004a,%r1,%v2,837,7,6

# Test immediate value range validation

# CHECK: error: unexpected operand type
# CHECK: .insn ri,0xa7080000,%r1,0x10000
        .insn ri,0xa7080000,%r1,0x10000        # 16-bit immediate, value too large (max 0xFFFF)

# CHECK: error: unexpected operand type
# CHECK: .insn ri,0xa7090000,%r1,0x8000
        .insn ri,0xa7090000,%r1,0x8000         # 16-bit signed immediate, value too large (max 0x7FFF)

# CHECK: error: unexpected operand type
# CHECK: .insn ri,0xa7090000,%r1,-0x8001
        .insn ri,0xa7090000,%r1,-0x8001        # 16-bit signed immediate, value too small (min -0x8000)

# CHECK: error: offset out of range
# CHECK: .insn ril,0xc20b00000000,%r2,0x100000000
        .insn ril,0xc20b00000000,%r2,0x100000000  # 32-bit immediate, value too large

# Test displacement range validation

# CHECK: error: unexpected operand type
# CHECK: .insn rx,0x5a000000,%r1,4096(%r2)
        .insn rx,0x5a000000,%r1,4096(%r2)      # 12-bit displacement, value too large (max 4095)

# CHECK: error: unexpected operand type
# CHECK: .insn rx,0x5a000000,%r1,-1(%r2)
        .insn rx,0x5a000000,%r1,-1(%r2)        # 12-bit unsigned displacement, negative value

# CHECK: error: unexpected operand type
# CHECK: .insn rxy,0xe30000000004,%r1,524288(%r2)
        .insn rxy,0xe30000000004,%r1,524288(%r2)  # 20-bit displacement, value too large (max 524287)

# CHECK: error: unexpected operand type
# CHECK: .insn rxy,0xe30000000004,%r1,-524289(%r2)
        .insn rxy,0xe30000000004,%r1,-524289(%r2) # 20-bit signed displacement, value too small (min -524288)

# Test mask/field range validation

# CHECK: error: unexpected operand type
# CHECK: .insn rie_c,0xec000000007e,%r1,256,2,0
        .insn rie_c,0xec000000007e,%r1,256,2,0    # 8-bit immediate field, value too large (max 255)

# CHECK: error: unexpected operand type
# CHECK: .insn rie_f,0xec000000005d,%r1,%r2,256,202,222
        .insn rie_f,0xec000000005d,%r1,%r2,256,202,222  # 8-bit field, value too large

# CHECK: error: unexpected operand type
# CHECK: .insn rie_f,0xec000000005d,%r1,%r2,250,256,222
        .insn rie_f,0xec000000005d,%r1,%r2,250,256,222  # 8-bit field, value too large

# CHECK: error: unexpected operand type
# CHECK: .insn rie_f,0xec000000005d,%r1,%r2,250,202,256
        .insn rie_f,0xec000000005d,%r1,%r2,250,202,256  # 8-bit field, value too large

# Test signed vs unsigned immediate validation

# CHECK: error: unexpected operand type
# CHECK: .insn rilu,0xc20b00000000,%r2,-1
        .insn rilu,0xc20b00000000,%r2,-1           # rilu expects unsigned 32-bit, negative value rejected

# CHECK: error: unexpected operand type
# CHECK: .insn sil,0xe56000000000,160(%r15),-32769
        .insn sil,0xe56000000000,160(%r15),-32769  # sil expects unsigned 16-bit, value too negative
