# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax %s \
# RUN:     | llvm-readobj -r - | FileCheck %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s \
# RUN:     | llvm-readobj -r - | FileCheck %s --check-prefix=RELAX

## Check that subtraction expressions emit R_LARCH_32_PCREL and R_LARCH_64_PCREL relocations.

## TODO: 1- or 2-byte data relocations are not supported for now.

# CHECK:      Relocations [
# CHECK-NEXT:     Section ({{.*}}) .rela.sx {
# CHECK-NEXT:       0x4 R_LARCH_PCALA_HI20 z 0x0
# CHECK-NEXT:       0x8 R_LARCH_PCALA_LO12 z 0x0
# CHECK-NEXT:       0xC R_LARCH_32_PCREL .sy 0x10
# CHECK-NEXT:     }
# CHECK-NEXT:     Section ({{.*}}) .rela.data {
# CHECK-NEXT:       0x0 R_LARCH_64_PCREL .sx 0x4
# CHECK-NEXT:       0x8 R_LARCH_64_PCREL .sy 0x8
# CHECK-NEXT:       0x10 R_LARCH_32_PCREL .sx 0x4
# CHECK-NEXT:       0x14 R_LARCH_32_PCREL .sy 0x8
# CHECK-NEXT:       0x18 R_LARCH_ADD64 .sx 0x4
# CHECK-NEXT:       0x18 R_LARCH_SUB64 .sy 0x8
# CHECK-NEXT:       0x20 R_LARCH_ADD64 .sy 0x8
# CHECK-NEXT:       0x20 R_LARCH_SUB64 .sx 0x4
# CHECK-NEXT:       0x28 R_LARCH_ADD32 .sx 0x4
# CHECK-NEXT:       0x28 R_LARCH_SUB32 .sy 0x8
# CHECK-NEXT:       0x2C R_LARCH_ADD32 .sy 0x8
# CHECK-NEXT:       0x2C R_LARCH_SUB32 .sx 0x4
# CHECK-NEXT:       0x30 R_LARCH_ADD64 .data 0x30
# CHECK-NEXT:       0x30 R_LARCH_SUB64 .sx 0x4
# CHECK-NEXT:       0x38 R_LARCH_ADD32 .data 0x38
# CHECK-NEXT:       0x38 R_LARCH_SUB32 .sy 0x8
# CHECK-NEXT:     }
# CHECK-NEXT:     Section ({{.*}}) .rela.sy {
# CHECK-NEXT:       0x0 R_LARCH_CALL36 foo 0x0
# CHECK-NEXT:       0x10 R_LARCH_32_PCREL .sx 0xC
# CHECK-NEXT:     }
# CHECK-NEXT:   ]

# RELAX:      Relocations [
# RELAX-NEXT:   Section ({{.*}}) .rela.sx {
# RELAX-NEXT:     0x4 R_LARCH_PCALA_HI20 z 0x0
# RELAX-NEXT:     0x4 R_LARCH_RELAX - 0x0
# RELAX-NEXT:     0x8 R_LARCH_PCALA_LO12 z 0x0
# RELAX-NEXT:     0x8 R_LARCH_RELAX - 0x0
# RELAX-NEXT:     0xC R_LARCH_ADD32 y 0x0
# RELAX-NEXT:     0xC R_LARCH_SUB32 x 0x0
# RELAX-NEXT:   }
# RELAX-NEXT:   Section ({{.*}}) .rela.data {
# RELAX-NEXT:     0x0 R_LARCH_64_PCREL x 0x0
# RELAX-NEXT:     0x8 R_LARCH_64_PCREL y 0x0
# RELAX-NEXT:     0x10 R_LARCH_32_PCREL x 0x0
# RELAX-NEXT:     0x14 R_LARCH_32_PCREL y 0x0
# RELAX-NEXT:     0x18 R_LARCH_ADD64 x 0x0
# RELAX-NEXT:     0x18 R_LARCH_SUB64 y 0x0
# RELAX-NEXT:     0x20 R_LARCH_ADD64 y 0x0
# RELAX-NEXT:     0x20 R_LARCH_SUB64 x 0x0
# RELAX-NEXT:     0x28 R_LARCH_ADD32 x 0x0
# RELAX-NEXT:     0x28 R_LARCH_SUB32 y 0x0
# RELAX-NEXT:     0x2C R_LARCH_ADD32 y 0x0
# RELAX-NEXT:     0x2C R_LARCH_SUB32 x 0x0
# RELAX-NEXT:     0x30 R_LARCH_ADD64 {{.*}} 0x0
# RELAX-NEXT:     0x30 R_LARCH_SUB64 x 0x0
# RELAX-NEXT:     0x38 R_LARCH_ADD32 {{.*}} 0x0
# RELAX-NEXT:     0x38 R_LARCH_SUB32 y 0x0
# RELAX-NEXT:   }
# RELAX-NEXT:   Section ({{.*}}) .rela.sy {
# RELAX-NEXT:     0x0 R_LARCH_CALL36 foo 0x0
# RELAX-NEXT:     0x0 R_LARCH_RELAX - 0x0
# RELAX-NEXT:     0x8 R_LARCH_ALIGN - 0xC
# RELAX-NEXT:     0x14 R_LARCH_ADD32 x 0x0
# RELAX-NEXT:     0x14 R_LARCH_SUB32 y 0x0
# RELAX-NEXT:   }
# RELAX-NEXT: ]

.section .sx,"ax"
nop
x:
la.pcrel $a0, z
.4byte y-x

.data
.8byte x-.
.8byte y-.
.4byte x-.
.4byte y-.
.8byte x-y
.8byte y-x
.4byte x-y
.4byte y-x
.8byte .-x
.4byte .-y

.section .sy,"ax"
call36 foo
y:
.p2align 4
.4byte x-y
