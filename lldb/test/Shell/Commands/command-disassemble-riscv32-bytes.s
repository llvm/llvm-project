# REQUIRES: riscv

# This test verifies that disassemble -b prints out the correct bytes and
# format for standard and unknown riscv instructions of various sizes,
# and that unknown instructions show opcodes and disassemble as "<unknown>".
# It also tests that the fdis command from examples/python/filter_disasm.py
# pipes the disassembly output through a simple filter program correctly.


# RUN: llvm-mc -filetype=obj -mattr=+c --triple=riscv32-unknown-unknown %s -o %t
# RUN: %lldb -b %t -o "command script import %S/../../../examples/python/filter_disasm.py" -o "fdis set %S/Inputs/dis_filt.sh" -o "fdis -n main" | FileCheck %s

main:
    addi   sp, sp, -0x20               # 16 bit standard instruction
    sw     a0, -0xc(s0)                # 32 bit standard instruction
    .insn 8, 0x2000200940003F;         # 64 bit custom instruction
    .insn 6, 0x021F | 0x00001000 << 32 # 48 bit xqci.e.li rd=8 imm=0x1000
    .insn 4, 0x84F940B                 # 32 bit xqci.insbi  
    .insn 2, 0xB8F2                    # 16 bit cm.push

# CHECK: Disassembly filter command (fdis) loaded
# CHECK: Fake filter start
# CHECK: [0x0] <+0>:   1101                     addi   sp, sp, -0x20 
# CHECK: [0x2] <+2>:   fea42a23                 sw     a0, -0xc(s0)
# CHECK: [0x6] <+6>:   0940003f 00200020        <unknown>
# CHECK: [0xe] <+14>:  021f 0000 1000           <unknown>
# CHECK: [0x14] <+20>: 084f940b                 <unknown>
# CHECK: [0x18] <+24>: b8f2                     <unknown>
# CHECK: Fake filter end

