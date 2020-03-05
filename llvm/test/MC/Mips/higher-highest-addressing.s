# RUN: llvm-mc -filetype=obj -triple=mips64el-unknown-linux -mcpu=mips64r2 %s \
# RUN:   | llvm-objdump -disassemble - | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=mips64el-unknown-linux -mcpu=mips64r2 %s \
# RUN:   | llvm-readobj -r | FileCheck %s -check-prefix=CHECK-REL


# Test that R_MIPS_HIGHER and R_MIPS_HIGHEST relocations are created.  By using
# NEXT we also test that none of the expressions from the test2 generates
# relocations.

test1:
# CHECK-LABEL:    <test1>:

        lui     $5, %highest(func)
        daddiu  $5, $5, %higher(func)

# CHECK-REL:           Relocations [
# CHECK-REL-NEXT:      {
# CHECK-REL-NEXT:          0x{{[0-9,A-F]+}} R_MIPS_HIGHEST
# CHECK-REL-NEXT:          0x{{[0-9,A-F]+}} R_MIPS_HIGHER
# CHECK-REL-NEXT:      }


# Test the calculation of %higher and %highest:
# ((x + 0x80008000) >> 32) & 0xffff (higher)
# ((x + 0x800080008000) >> 48) & 0xffff (highest).

test2:
# CHECK-LABEL:    <test2>:

# Check the case where relocations are not modified by adding +1.  The constant
# is chosen so that it is just below the value that triggers the addition of +1
# to %higher.
$L1:
        lui     $6,    %highest($L2-$L1+0x300047FFF7FF7)
        daddiu  $6, $6, %higher($L2-$L1+0x300047FFF7FF7)
$L2:
# CHECK:    lui      $6, 3
# CHECK:    daddiu   $6, $6, 4


# Check the case where %higher is modified by adding +1.
        lui     $7, %highest($L2-$L1+0x300047FFF7FF8)
        ld      $7, %higher ($L2-$L1+0x300047FFF7FF8)($7)
# CHECK:    lui     $7, 3
# CHECK:    ld      $7, 5($7)


# Check the case where both %higher and %highest are modified by adding +1.
        lui     $8, %highest(0x37FFF7FFF8000)
        ld      $8, %higher (0x37FFF7FFF8000)($8)
# CHECK:    lui     $8, 4
# CHECK:    ld      $8, -32768($8)
