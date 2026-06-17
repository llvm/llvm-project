## Test that llvm-objdump uses ISA mapping symbols ("$x<ISAString>") to
## configure per-region disassembly.  Covers:
##   - Initial non-V region: V-encoded bytes appear as <unknown>
##   - V-enabled region: same bytes correctly decode as vadd.vv
##   - Restored non-V region (after .option pop): back to <unknown>
##   - Second V region: verifies ISA target cache is reused correctly
##   - rv32 and rv64 both exercise parseNormalizedArchString

# RUN: llvm-mc -triple=riscv64 -mattr=+c -filetype=obj %s -o %t.64.o
# RUN: llvm-objdump -d --no-show-raw-insn %t.64.o | FileCheck %s
#
# RUN: llvm-mc -triple=riscv32 -mattr=+c -filetype=obj %s -o %t.32.o
# RUN: llvm-objdump -d --no-show-raw-insn %t.32.o | FileCheck %s

.text

# Before any .option arch change the active ISA is rv64gc / rv32gc.
# A regular instruction decodes normally; a V-encoded word is <unknown>.
nop
# CHECK:      0:      	nop
.insn 4, 0x02110057  # vadd.vv v0, v1, v2
# CHECK-NEXT: 2:      	<unknown>

## Region 1: V extension enabled via .option arch, +v.
## The assembler emits "$x<ISA+V>" before vadd.vv; objdump must decode it
## with the V decoder active.
.option push
.option arch, +v
vadd.vv v0, v1, v2
# CHECK-NEXT: 6:      	vadd.vv	v0, v1, v2
.option pop

## Restored to rv64gc / rv32gc after pop: same encoding is <unknown> again.
.insn 4, 0x02110057  # vadd.vv v0, v1, v2
# CHECK-NEXT: a:      	<unknown>

## Region 2: enter V again.  This exercises the ISATargetCache reuse path
## (the DisassemblerTarget for ISA+V was already created in Region 1).
.option push
.option arch, +v
vadd.vv v8, v9, v10
# CHECK-NEXT: e:      	vadd.vv	v8, v9, v10
.option pop

## Back to base ISA once more: V encoding is <unknown>.
.insn 4, 0x02110057  # vadd.vv v0, v1, v2
# CHECK-NEXT: 12:      	<unknown>
