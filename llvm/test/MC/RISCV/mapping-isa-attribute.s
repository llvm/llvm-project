## Instruction mapping symbols with ISA string

# RUN: llvm-mc -triple=riscv32 -filetype=obj -mattr=+c -o %t.o %s
# RUN: llvm-objdump -t %t.o | FileCheck %s -check-prefix=CHECK-MAPPINGSYMBOLS
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+c -o %t.o %s
# RUN: llvm-objdump -t %t.o | FileCheck %s -check-prefix=CHECK-MAPPINGSYMBOLS

.text
.attribute arch, "rv32ic"
nop
# CHECK-MAPPINGSYMBOLS-NOT: $xrv32i2p1_c2p0
# No mapping symbol as arch has not changed given the attributes

.attribute arch, "rv32i"
# CHECK-MAPPINGSYMBOLS: $xrv32i2p1

.attribute arch, "rv32i"
nop
nop
# CHECK-MAPPINGSYMBOLS-NOT: $xrv32i2p1
## Multiple instructions with same isa string, so no mapping symbol expected.

.word 4
nop
# CHECK-MAPPINGSYMBOLS-NOT: $xrv32i2p1
# CHECK-MAPPINGSYMBOLS: $x
## Data followed by an instruction should produce an instruction mapping
## symbol, but the isa string should not be present.

.attribute arch, "rv32i2p1"
nop
# CHECK-MAPPINGSYMBOLS-NOT: $xrv32i2p0
## The arch "rv32ic" and "rv32i2p1" has the same isa string, so no mapping
## symbol expected.

.attribute arch, "rv32e"
nop
# CHECK-MAPPINGSYMBOLS: $xrv32e2p0

.attribute arch, "rv64e"
nop
# CHECK-MAPPINGSYMBOLS: $xrv64e2p0

.attribute arch, "rv32g"
nop
# CHECK-MAPPINGSYMBOLS: $xrv32i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0_zifencei2p0

.attribute arch, "rv64g"
nop
# CHECK-MAPPINGSYMBOLS: $xrv64i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0_zifencei2p0
