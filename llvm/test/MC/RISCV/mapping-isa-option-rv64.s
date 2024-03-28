## Instruction mapping symbols with ISA string

# RUN: llvm-mc -triple=riscv64 -mattr=+f,+a -filetype=obj -o %t.o %s
# RUN: llvm-objdump -t %t.o | FileCheck %s -check-prefix=CHECK-MAPPINGSYMBOLS

.text
nop
# CHECK-MAPPINGSYMBOLS-NOT: $xrv64

.option push
.option arch, -f
nop
# CHECK-MAPPINGSYMBOLS: $xrv64i2p1_a2p1_zicsr2p0

.option pop
nop
# CHECK-MAPPINGSYMBOLS: $xrv64i2p1_a2p1_f2p2_zicsr2p0

.option push
.option arch, +c
nop
# CHECK-MAPPINGSYMBOLS: $xrv64i2p1_a2p1_f2p2_c2p0_zicsr2p0

.option pop
nop
# CHECK-MAPPINGSYMBOLS: $xrv64i2p1_a2p1_f2p2_zicsr2p0

.option arch, rv64imac
nop
# CHECK-MAPPINGSYMBOLS: $xrv64i2p1_m2p0_a2p1_c2p0
nop
# CHECK-MAPPINGSYMBOLS-NOT: $xrv64i2p1_m2p0_a2p1_c2p0

.word 4
nop
# CHECK-MAPPINGSYMBOLS-NOT: $xrv64i2p1_m2p0_a2p1_c2p0
