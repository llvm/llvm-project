# RUN: not llvm-mc -triple=hexagon -filetype=asm %s 2>&1 \
# RUN:   | FileCheck %s --implicit-check-not="error:"

# Test coverage for HexagonMCChecker: exercise various packet constraint
# violations that the checker reports.

# --- Multiple writes to the same register ---
# CHECK: error: register `R0' modified more than once
{
  r0 = r1
  r0 = r2
}

# --- Read-only register PC ---
# CHECK: error: Cannot write to read-only register `PC'
{
  pc = r0
}

# --- New-value register consumer has no producer ---
# CHECK: error: New value register consumer has no producer
{
  if (cmp.eq(r0.new, #0)) jump:nt .Ltmp
}
.Ltmp:
  nop
