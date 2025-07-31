# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilb %t/pass.s -o - \
# RUN:   | llvm-objdump -dr -M no-aliases - --mattr=+experimental-xqcilb | FileCheck %s
# RUN: not llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilb %t/fail.s \
# RUN:   2>&1 | FileCheck %t/fail.s --check-prefix=ERROR

## This testcase shows how `c.j`, `c.jal` and `jal` can be relaxed to `qc.e.j` and `qc.e.jal`
## with Xqcilb, when the branches are out of range, but also that these can be compressed
## when referencing close labels.

## The only problem we have here is when `jal` references an out-of-range label, and `rd` is
## not `x0` or `x1` - for which we have no equivalent sequence, so we just fail to relax, and
## emit a fixup-value-out-of-range error.

#--- pass.s

EXT_JUMP_NEGATIVE:
  c.nop
.space 0x100000

FAR_JUMP_NEGATIVE:
  c.nop
.space 0x1000

NEAR_NEGATIVE:
  c.nop

start:
  c.j NEAR
# CHECK: c.j {{0x[0-9a-f]+}} <NEAR>
  c.j NEAR_NEGATIVE
# CHECK: c.j {{0x[0-9a-f]+}} <NEAR_NEGATIVE>
  c.j FAR_JUMP
# CHECK: jal zero, {{0x[0-9a-f]+}} <FAR_JUMP>
  c.j FAR_JUMP_NEGATIVE
# CHECK: jal zero, {{0x[0-9a-f]+}} <FAR_JUMP_NEGATIVE>
  c.j EXT_JUMP
# CHECK: qc.e.j {{0x[0-9a-f]+}} <EXT_JUMP>
  c.j EXT_JUMP_NEGATIVE
# CHECK: qc.e.j {{0x[0-9a-f]+}} <EXT_JUMP_NEGATIVE>
  c.j undef
# CHECK: qc.e.j {{0x[0-9a-f]+}} <start+{{0x[0-9a-f]+}}>
# CHECK: R_RISCV_CUSTOM195 undef

  c.jal NEAR
# CHECK: c.jal {{0x[0-9a-f]+}} <NEAR>
  c.jal NEAR_NEGATIVE
# CHECK: c.jal {{0x[0-9a-f]+}} <NEAR_NEGATIVE>
  c.jal FAR_JUMP
# CHECK: jal ra, {{0x[0-9a-f]+}} <FAR_JUMP>
  c.jal FAR_JUMP_NEGATIVE
# CHECK: jal ra, {{0x[0-9a-f]+}} <FAR_JUMP_NEGATIVE>
  c.jal EXT_JUMP
# CHECK: qc.e.jal {{0x[0-9a-f]+}} <EXT_JUMP>
  c.jal EXT_JUMP_NEGATIVE
# CHECK: qc.e.jal {{0x[0-9a-f]+}} <EXT_JUMP_NEGATIVE>
  c.jal undef
# CHECK: qc.e.jal {{0x[0-9a-f]+}} <start+{{0x[0-9a-f]+}}>
# CHECK: R_RISCV_CUSTOM195 undef

  jal zero, NEAR
# CHECK: c.j {{0x[0-9a-f]+}} <NEAR>
  jal zero, NEAR_NEGATIVE
# CHECK: c.j {{0x[0-9a-f]+}} <NEAR_NEGATIVE>
  jal zero, FAR_JUMP
# CHECK: jal zero, {{0x[0-9a-f]+}} <FAR_JUMP>
  jal zero, FAR_JUMP_NEGATIVE
# CHECK: jal zero, {{0x[0-9a-f]+}} <FAR_JUMP_NEGATIVE>
  jal zero, EXT_JUMP
# CHECK: qc.e.j {{0x[0-9a-f]+}} <EXT_JUMP>
  jal zero, EXT_JUMP_NEGATIVE
# CHECK: qc.e.j {{0x[0-9a-f]+}} <EXT_JUMP_NEGATIVE>
  jal zero, undef
# CHECK: qc.e.j {{0x[0-9a-f]+}} <start+{{0x[0-9a-f]+}}>
# CHECK: R_RISCV_CUSTOM195 undef

  jal ra, NEAR
# CHECK: c.jal {{0x[0-9a-f]+}} <NEAR>
  jal ra, NEAR_NEGATIVE
# CHECK: c.jal {{0x[0-9a-f]+}} <NEAR_NEGATIVE>
  jal ra, FAR_JUMP
# CHECK: jal ra, {{0x[0-9a-f]+}} <FAR_JUMP>
  jal ra, FAR_JUMP_NEGATIVE
# CHECK: jal ra, {{0x[0-9a-f]+}} <FAR_JUMP_NEGATIVE>
  jal ra, EXT_JUMP
# CHECK: qc.e.jal {{0x[0-9a-f]+}} <EXT_JUMP>
  jal ra, EXT_JUMP_NEGATIVE
# CHECK: qc.e.jal {{0x[0-9a-f]+}} <EXT_JUMP_NEGATIVE>
  jal ra, undef
# CHECK: qc.e.jal {{0x[0-9a-f]+}} <start+{{0x[0-9a-f]+}}>
# CHECK: R_RISCV_CUSTOM195 undef

  qc.e.j NEAR
# CHECK: c.j {{0x[0-9a-f]+}} <NEAR>
  qc.e.j NEAR_NEGATIVE
# CHECK: c.j {{0x[0-9a-f]+}} <NEAR_NEGATIVE>
  qc.e.j FAR_JUMP
# CHECK: jal zero, {{0x[0-9a-f]+}} <FAR_JUMP>
  qc.e.j FAR_JUMP_NEGATIVE
# CHECK: jal zero, {{0x[0-9a-f]+}} <FAR_JUMP_NEGATIVE>
  qc.e.j EXT_JUMP
# CHECK: qc.e.j {{0x[0-9a-f]+}} <EXT_JUMP>
  qc.e.j EXT_JUMP_NEGATIVE
# CHECK: qc.e.j {{0x[0-9a-f]+}} <EXT_JUMP_NEGATIVE>
  qc.e.j undef
# CHECK: qc.e.j {{0x[0-9a-f]+}} <start+{{0x[0-9a-f]+}}>
# CHECK: R_RISCV_CUSTOM195 undef

  qc.e.jal NEAR
# CHECK: c.jal {{0x[0-9a-f]+}} <NEAR>
  qc.e.jal NEAR_NEGATIVE
# CHECK: c.jal {{0x[0-9a-f]+}} <NEAR_NEGATIVE>
  qc.e.jal FAR_JUMP
# CHECK: jal ra, {{0x[0-9a-f]+}} <FAR_JUMP>
  qc.e.jal FAR_JUMP_NEGATIVE
# CHECK: jal ra, {{0x[0-9a-f]+}} <FAR_JUMP_NEGATIVE>
  qc.e.jal EXT_JUMP
# CHECK: qc.e.jal {{0x[0-9a-f]+}} <EXT_JUMP>
  qc.e.jal EXT_JUMP_NEGATIVE
# CHECK: qc.e.jal {{0x[0-9a-f]+}} <EXT_JUMP_NEGATIVE>
  qc.e.jal undef
# CHECK: qc.e.jal {{0x[0-9a-f]+}} <start+{{0x[0-9a-f]+}}>
# CHECK: R_RISCV_CUSTOM195 undef



  jal t1, NEAR
# CHECK: jal t1, {{0x[0-9a-f]+}} <NEAR>
  jal t1, NEAR_NEGATIVE
# CHECK: jal t1, {{0x[0-9a-f]+}} <NEAR_NEGATIVE>
  jal t1, FAR_JUMP
# CHECK: jal t1, {{0x[0-9a-f]+}} <FAR_JUMP>
  jal t1, FAR_JUMP_NEGATIVE
# CHECK: jal t1, {{0x[0-9a-f]+}} <FAR_JUMP_NEGATIVE>

## The two cases with EXT_JUMP and EXT_JUMP_NEGATIVE are
## in fail.s, below.

  jal t1, undef
# CHECK: jal t1, {{0x[0-9a-f]+}} <start+{{0x[0-9a-f]+}}>
# CHECK: R_RISCV_JAL undef


NEAR:
  c.nop
.space 0x1000
FAR_JUMP:
  c.nop
.space 0x100000
EXT_JUMP:
  c.nop


#--- fail.s


EXT_JUMP_NEGATIVE:
  c.nop
.space 0x100000
.space 0x1000

  jal t1, EXT_JUMP
# ERROR: [[@LINE-1]]:11: error: fixup value out of range
  jal t1, EXT_JUMP_NEGATIVE
# ERROR: [[@LINE-1]]:11: error: fixup value out of range

.space 0x1000
.space 0x100000
EXT_JUMP:
  c.nop
