## Test llvm-readobj & llvm-readelf can decode RISC-V attributes correctly.

# RUN: llvm-mc -triple riscv32 -filetype obj -o %t.rv32.o %s
# RUN: llvm-mc -triple riscv64 -filetype obj -o %t.rv64.o %s
# RUN: llvm-readobj --arch-specific %t.rv32.o \
# RUN:   | FileCheck %s --check-prefix=CHECK-OBJ
# RUN: llvm-readelf -A %t.rv32.o \
# RUN:   | FileCheck %s --check-prefix=CHECK-OBJ
# RUN: llvm-readobj --arch-specific %t.rv64.o \
# RUN:   | FileCheck %s --check-prefix=CHECK-OBJ
# RUN: llvm-readelf -A %t.rv64.o \
# RUN:   | FileCheck %s --check-prefix=CHECK-OBJ

.attribute  Tag_stack_align, 16
# CHECK-OBJ:      Tag: 4
# CHECK-OBJ-NEXT: Value: 16
# CHECK-OBJ-NEXT: TagName: stack_align
# CHECK-OBJ-NEXT: Description: Stack alignment is 16-bytes

.attribute  Tag_arch, "rv32i2p0_m2p0_a2p0_c2p0"
# CHECK-OBJ:      Tag: 5
# CHECK-OBJ-NEXT: TagName: arch
# CHECK-OBJ-NEXT: Value: rv32i2p0_m2p0_a2p0_c2p0

.attribute  Tag_unaligned_access, 0
# CHECK-OBJ:      Tag: 6
# CHECK-OBJ-NEXT: Value: 0
# CHECK-OBJ-NEXT: TagName: unaligned_access
# CHECK-OBJ-NEXT: Description: No unaligned access

.attribute  Tag_priv_spec, 2
# CHECK-OBJ:      Tag: 8
# CHECK-OBJ-NEXT: TagName: priv_spec
# CHECK-OBJ-NEXT: Value: 2

.attribute  Tag_priv_spec_minor, 0
# CHECK-OBJ:      Tag: 10
# CHECK-OBJ-NEXT: TagName: priv_spec_minor
# CHECK-OBJ-NEXT: Value: 0

.attribute  Tag_priv_spec_revision, 0
# CHECK-OBJ:      Tag: 12
# CHECK-OBJ-NEXT: TagName: priv_spec_revision
# CHECK-OBJ-NEXT: Value: 0
