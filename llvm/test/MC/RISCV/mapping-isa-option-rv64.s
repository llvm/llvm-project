## Test ISA mapping symbol emission for .option rvc/norvc/arch/push/pop on rv64.
## The assembler emits a "$x<ISAString>" symbol when the active ISA changes,
## using lazy emission (the symbol appears before the first instruction after
## the change) and deduplication (no symbol when the ISA is unchanged).
## An initial "$x<ISAString>" symbol is also emitted before the very first
## instruction to record the starting ISA in the object file.

# RUN: llvm-mc -triple=riscv64 -mattr=+f,+a -filetype=obj -o %t.o %s
# RUN: llvm-readobj --symbols %t.o | FileCheck %s

.text
nop
# The initial ISA mapping symbol is emitted before the first instruction.
# CHECK: Name: $xrv64i2p1_a2p1_f2p2_zicsr2p0_zaamo1p0_zalrsc1p0
# CHECK-NEXT: Value: 0x0

# .option rvc enables C; a new symbol is emitted before the next instruction.
.option rvc
nop
# CHECK: Name: $xrv64i2p1_a2p1_f2p2_c2p0_zicsr2p0_zaamo1p0_zalrsc1p0_zca1p0
# CHECK-NEXT: Value: 0x4

# .option norvc disables C; a new symbol without C is emitted.
.option norvc
nop
# CHECK: Name: $xrv64i2p1_a2p1_f2p2_zicsr2p0_zaamo1p0_zalrsc1p0
# CHECK-NEXT: Value: 0x6

# .option push saves the current ISA; .option arch switches to a full ISA;
# .option pop restores the pre-push ISA and emits a new symbol for it.
.option push
.option arch, rv64imac
nop
# CHECK: Name: $xrv64i2p1_m2p0_a2p1_c2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0
# CHECK-NEXT: Value: 0xA

.option pop
nop
# CHECK: Name: $xrv64i2p1_a2p1_f2p2_zicsr2p0_zaamo1p0_zalrsc1p0
# CHECK-NEXT: Value: 0xC

# Deduplication: repeating the same .option arch does not emit a second symbol.
.option arch, rv64imac
nop
# CHECK: Name: $xrv64i2p1_m2p0_a2p1_c2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0
# CHECK-NEXT: Value: 0x10
.option arch, rv64imac
nop
# No additional symbol expected between the two instructions above.
# CHECK-NOT: $xrv64i2p1_m2p0_a2p1_c2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0 {{.*}}Value: 0x1[^0]

# Nested push/pop: walk down to an inner ISA via two pushes with distinct
# .option arch in between, then walk back up with two pops.  Each level must
# emit its own mapping symbol so the stack is observable in the object.
.option push
.option arch, rv64im
nop
# CHECK: Name: $xrv64i2p1_m2p0_zmmul1p0
# CHECK-NEXT: Value: 0x14

.option push
.option arch, rv64imc
nop
# CHECK: Name: $xrv64i2p1_m2p0_c2p0_zmmul1p0_zca1p0
# CHECK-NEXT: Value: 0x18

.option pop
nop
# CHECK: Name: $xrv64i2p1_m2p0_zmmul1p0
# CHECK-NEXT: Value: 0x1A

.option pop
nop
# CHECK: Name: $xrv64i2p1_m2p0_a2p1_c2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0
# CHECK-NEXT: Value: 0x1E
