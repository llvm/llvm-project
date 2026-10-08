# RUN: not llvm-mc -triple=hexagon -filetype=asm %s 2>&1 \
# RUN:   | FileCheck %s --implicit-check-not="error:"

# Test coverage for HexagonMCChecker: exercise HW loop + branch conflict
# and out-of-slots errors.

# --- HW loop + branch conflict ---
# When :endloop0 is present, branching instructions are not allowed.
# CHECK: error: Branches cannot be in a packet with hardware loops
{
  r0 = r1
  jump .Ltarget
}:endloop0
.Ltarget:

# --- Out of slots: too many instructions in a packet ---
# CHECK: error: invalid instruction packet: out of slots
{
  r0 = r1
  r2 = r3
  r4 = r5
  r6 = r7
  r8 = r9
}
