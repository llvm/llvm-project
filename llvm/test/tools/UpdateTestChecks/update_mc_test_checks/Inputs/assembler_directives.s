// RUN: llvm-mc -triple=riscv32 -I %S %s | FileCheck --check-prefixes=CHECK,CHECK-32 %s
// RUN: llvm-mc -triple=riscv64 -I %S --defsym=RV64=1 %s | FileCheck --check-prefixes=CHECK,CHECK-64 %s

/// Check that instructions inside .ifdef/.else are correctly checked with their respective prefixes
.ifdef RV64
ld a0, 0(a1)
.else
lw a0, 0(a1)
.endif

/// The macro definition itself should not get check lines
.macro load_reg reg, addr
  lw \reg, 0(\addr)
  sw \reg, 0(\addr)
.endm

/// Macro instantiations should get check lines
load_reg a0, a1

/// We should not add check lines for instructions originating from .include files
.include "include_file.inc"
