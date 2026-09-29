# RUN: llvm-mc -triple=hexagon -filetype=obj %s \
# RUN:   | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

# Test coverage for HexagonAsmParser: exercise vrcmpys instruction
# transformations with odd/even register mapping.

# --- M2_vrcmpys_s1 odd register -> hi ---
# CHECK-LABEL: <test_vrcmpys_hi>:
# CHECK: vrcmpys(r3:2,r1:0):<<1:sat:raw:hi
.globl test_vrcmpys_hi
test_vrcmpys_hi:
  r1:0 = vrcmpys(r3:2, r1):<<1:sat
  jumpr lr

# --- M2_vrcmpys_s1 even register -> lo ---
# CHECK-LABEL: <test_vrcmpys_lo>:
# CHECK: vrcmpys(r3:2,r1:0):<<1:sat:raw:lo
.globl test_vrcmpys_lo
test_vrcmpys_lo:
  r1:0 = vrcmpys(r3:2, r0):<<1:sat
  jumpr lr

# --- M2_vrcmpys_acc_s1 with odd register -> hi accumulate ---
# CHECK-LABEL: <test_vrcmpys_acc_hi>:
# CHECK: += vrcmpys(r5:4,r1:0):<<1:sat:raw:hi
.globl test_vrcmpys_acc_hi
test_vrcmpys_acc_hi:
  r1:0 += vrcmpys(r5:4, r1):<<1:sat
  jumpr lr

# --- M2_vrcmpys_acc_s1 with even register -> lo accumulate ---
# CHECK-LABEL: <test_vrcmpys_acc_lo>:
# CHECK: += vrcmpys(r5:4,r1:0):<<1:sat:raw:lo
.globl test_vrcmpys_acc_lo
test_vrcmpys_acc_lo:
  r1:0 += vrcmpys(r5:4, r0):<<1:sat
  jumpr lr

# --- M2_vrcmpys_s1rp with odd register -> rnd:sat:hi ---
# CHECK-LABEL: <test_vrcmpys_rp_hi>:
# CHECK: vrcmpys(r3:2,r1:0):<<1:rnd:sat:raw:hi
.globl test_vrcmpys_rp_hi
test_vrcmpys_rp_hi:
  r0 = vrcmpys(r3:2, r1):<<1:rnd:sat
  jumpr lr

# --- M2_vrcmpys_s1rp with even register -> rnd:sat:lo ---
# CHECK-LABEL: <test_vrcmpys_rp_lo>:
# CHECK: vrcmpys(r3:2,r1:0):<<1:rnd:sat:raw:lo
.globl test_vrcmpys_rp_lo
test_vrcmpys_rp_lo:
  r0 = vrcmpys(r3:2, r0):<<1:rnd:sat
  jumpr lr
