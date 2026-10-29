# RUN: llvm-mc -triple=hexagon -filetype=obj %s \
# RUN:   | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

# Test coverage for HexagonAsmParser instruction transformations that are
# performed during parsing: table-index instructions, multiply signed
# immediate, arithmetic shift right with round, register pair transfers,
# and bounds check register mapping.

# --- Table-index instructions (goodsyntax -> real opcode) ---

# tableidxb: no immediate adjustment needed
# CHECK-LABEL: <test_tableidxb>:
# CHECK: tableidxb(r1,#4,#3):raw
.globl test_tableidxb
test_tableidxb:
  r0 = tableidxb(r1, #4, #3)
  jumpr lr

# tableidxh: subtracts 1 from width
# CHECK-LABEL: <test_tableidxh>:
# CHECK: tableidxh(r2,#3,#4):raw
.globl test_tableidxh
test_tableidxh:
  r0 = tableidxh(r2, #3, #5)
  jumpr lr

# tableidxw: subtracts 2 from width
# CHECK-LABEL: <test_tableidxw>:
# CHECK: tableidxw(r3,#2,#4):raw
.globl test_tableidxw
test_tableidxw:
  r0 = tableidxw(r3, #2, #6)
  jumpr lr

# tableidxd: subtracts 3 from width
# CHECK-LABEL: <test_tableidxd>:
# CHECK: tableidxd(r4,#1,#4):raw
.globl test_tableidxd
test_tableidxd:
  r0 = tableidxd(r4, #1, #7)
  jumpr lr

# --- Multiply signed immediate ---

# M2_mpysmi with positive value -> M2_mpysip
# CHECK-LABEL: <test_mpysmi_pos>:
# CHECK: r0 = +mpyi(r1,#10)
.globl test_mpysmi_pos
test_mpysmi_pos:
  r0 = mpyi(r1, #10)
  jumpr lr

# M2_mpysmi with negative value -> M2_mpysin
# CHECK-LABEL: <test_mpysmi_neg>:
# CHECK: r0 = -mpyi(r1,#10)
.globl test_mpysmi_neg
test_mpysmi_neg:
  r0 = -mpyi(r1, #10)
  jumpr lr

# M2_mpyui -> M2_mpyi
# CHECK-LABEL: <test_mpyui>:
# CHECK: r0 = mpyi(r1,r2)
.globl test_mpyui
test_mpyui:
  r0 = mpyui(r1, r2)
  jumpr lr

# --- Arithmetic shift right with round ---

# S2_asr_i_r_rnd_goodsyntax with shift==0 produces asr(r,#0):rnd
# CHECK-LABEL: <test_asr_rnd_zero>:
# CHECK: asr(r1,#0):rnd
.globl test_asr_rnd_zero
test_asr_rnd_zero:
  r0 = asr(r1, #0):rnd
  jumpr lr

# S2_asr_i_r_rnd with shift>0
# CHECK-LABEL: <test_asr_rnd_nonzero>:
# CHECK: asr(r1,#{{[0-9]+}}):rnd
.globl test_asr_rnd_nonzero
test_asr_rnd_nonzero:
  r0 = asr(r1, #4):rnd
  jumpr lr

# S2_asr_i_p_rnd with shift==0 -> A2_combinew
# CHECK-LABEL: <test_asr_p_rnd_zero>:
# CHECK: asr(r3:2,#0):rnd
.globl test_asr_p_rnd_zero
test_asr_p_rnd_zero:
  r1:0 = asr(r3:2, #0):rnd
  jumpr lr

# S2_asr_i_p_rnd with shift>0
# CHECK-LABEL: <test_asr_p_rnd_nonzero>:
# CHECK: asr(r3:2,#{{[0-9]+}}):rnd
.globl test_asr_p_rnd_nonzero
test_asr_p_rnd_nonzero:
  r1:0 = asr(r3:2, #8):rnd
  jumpr lr

# --- Register pair transfers ---

# A2_tfrp -> A2_combinew
# CHECK-LABEL: <test_tfrp>:
# CHECK: r1:0 = combine(r3,r2)
.globl test_tfrp
test_tfrp:
  r1:0 = r3:2
  jumpr lr

# A2_tfrpt -> C2_ccombinewt (predicated true pair transfer)
# CHECK-LABEL: <test_tfrpt>:
# CHECK: if (p0) r1:0 = combine(r3,r2)
.globl test_tfrpt
test_tfrpt:
  if (p0) r1:0 = r3:2
  jumpr lr

# A2_tfrpf -> C2_ccombinewf (predicated false pair transfer)
# CHECK-LABEL: <test_tfrpf>:
# CHECK: if (!p0) r1:0 = combine(r3,r2)
.globl test_tfrpf
test_tfrpf:
  if (!p0) r1:0 = r3:2
  jumpr lr

# A2_tfrptnew -> C2_ccombinewnewt
# CHECK-LABEL: <test_tfrptnew>:
# CHECK: if (p0.new) r1:0 = combine(r3,r2)
.globl test_tfrptnew
test_tfrptnew:
{
  p0 = cmp.eq(r10, r11)
  if (p0.new) r1:0 = r3:2
}
  jumpr lr

# A2_tfrpfnew -> C2_ccombinewnewf
# CHECK-LABEL: <test_tfrpfnew>:
# CHECK: if (!p0.new) r1:0 = combine(r3,r2)
.globl test_tfrpfnew
test_tfrpfnew:
{
  p0 = cmp.eq(r10, r11)
  if (!p0.new) r1:0 = r3:2
}
  jumpr lr

# --- Bounds check register mapping ---

# A4_boundscheck with even register -> A4_boundscheck_lo
# CHECK-LABEL: <test_boundscheck_even>:
# CHECK: boundscheck(r1:0,r3:2):raw:lo
.globl test_boundscheck_even
test_boundscheck_even:
  p0 = boundscheck(r0, r3:2)
  jumpr lr

# A4_boundscheck with odd register -> A4_boundscheck_hi
# CHECK-LABEL: <test_boundscheck_odd>:
# CHECK: boundscheck(r1:0,r3:2):raw:hi
.globl test_boundscheck_odd
test_boundscheck_odd:
  p0 = boundscheck(r1, r3:2)
  jumpr lr
