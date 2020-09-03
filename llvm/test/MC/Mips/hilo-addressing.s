# RUN: llvm-mc -show-encoding -triple mips-unknown-unknown %s \
# RUN:  | FileCheck %s -check-prefix=CHECK-ENC

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux %s \
# RUN:  | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INSTR

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux %s \
# RUN:  | llvm-readobj -r | FileCheck %s -check-prefix=CHECK-REL


# Check that 1 is added to the high 16 bits if bit 15 of the low part is 1.

        .equ    addr, 0xdeadbeef
        lui     $4, %hi(addr)
        lb      $2, %lo(addr)($4)
# CHECK-ENC: # encoding: [0x3c,0x04,0xde,0xae]
# CHECK-ENC: # encoding: [0x80,0x82,0xbe,0xef]


# Check that assembler can handle %hi(label1 - label2) and %lo(label1 - label2)
# expressions.

$L1:
        # Emit zeros so that difference between $L3 and $L1 is 0x30124 bytes.
        .fill 0x30124-8
$L2:
        lui     $4, %hi($L3-$L1)
        addiu   $4, $4, %lo($L3-$L1)
# CHECK-INSTR:    lui     $4, 3
# CHECK-INSTR:    addiu   $4, $4, 292

$L3:
        lui     $5, %hi($L2-$L3)
        lw      $5, %lo($L2-$L3)($5)
# CHECK-INSTR:    lui     $5, 0
# CHECK-INSTR:    lw      $5, -8($5)


# Check that relocation isn't emitted for %hi(label1 - label2) and
# %lo(label1 - label2) expressions.

# CHECK-REL-NOT:    R_MIPS
