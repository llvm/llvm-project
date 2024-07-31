# RUN: llvm-mc --filetype=obj --triple=loongarch64 %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

## Check that subtraction expressions emit R_LARCH_32_PCREL and R_LARCH_64_PCREL relocations.

## TODO: 1- or 2-byte data relocations are not supported for now.

# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.data {
# CHECK-NEXT:     0x0 R_LARCH_64_PCREL sx 0x0
# CHECK-NEXT:     0x8 R_LARCH_64_PCREL sy 0x0
# CHECK-NEXT:     0x10 R_LARCH_32_PCREL sx 0x0
# CHECK-NEXT:     0x14 R_LARCH_32_PCREL sy 0x0
# CHECK-NEXT:   }

.section sx,"a"
x:
nop

.data
.8byte x-.
.8byte y-.
.4byte x-.
.4byte y-.

.section sy,"a"
y:
nop
