// RUN: llvm-mc -triple=riscv32 -mattr=+f -defsym=RV64=0 --show-encoding --show-inst %s | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-32,CHECK-ASM-32 %s
// RUN: llvm-mc -triple=riscv64 -mattr=+f -defsym=RV64=1 --show-encoding --show-inst %s | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-64,CHECK-ASM-64 %s
// RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+f -defsym=RV64=0 < %s | llvm-objdump --mattr=-f -d - | FileCheck %s --check-prefixes=CHECK,CHECK-OBJ,CHECK-32,CHECK-OBJ-32
// RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+f -defsym=RV64=1 < %s | llvm-objdump --mattr=-f -d - | FileCheck %s --check-prefixes=CHECK,CHECK-OBJ,CHECK-64,CHECK-OBJ-64

/// Check that identical instructions across assembly and disassembly are merged under CHECK
add a0, a0, a1

/// Check target-specific instructions that differ completely depending on the mode
.if RV64
ld a0, 0(a1)
.else
lw a0, 0(a1)
.endif

/// Check instructions that have target-specific preprocessor immediate values depending on the mode
addi a0, a0, RV64

/// Check disassembly of an invalid instruction mapping to <unknown> (we disable F for llvm-objdump output)
fadd.s fa0, fa0, fa1
