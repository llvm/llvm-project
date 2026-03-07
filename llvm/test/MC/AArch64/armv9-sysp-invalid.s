// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+d128 < %s \
// RUN:   | llvm-objdump -d --mattr=+d128 --no-print-imm-hex - \
// RUN:   | FileCheck %s --check-prefix=DISASM

// Ensure invalid SYSP encodings are not printed as TLBIP aliases.

// op1 = 7 (outside architecturally valid SYSP range for op1)
// Cn = 8, Cm = 0, op2 = 0, Rt = 0 (x0, x1)
.inst 0xd54f8000

// Cn = 0 (outside architecturally valid SYSP range for Cn)
.inst 0xd5480000

// Cm = 8 (outside architecturally valid SYSP range for Cm)
.inst 0xd5488800

// DISASM-NOT: tlbip
// DISASM: <unknown>
// DISASM: <unknown>
// DISASM: <unknown>
