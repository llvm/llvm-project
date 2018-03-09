// Check generation of N32 ABI relocations.

// RUN: llvm-mc -filetype=obj -triple=mips64-linux-gnu -mcpu=mips3 \
// RUN:         -target-abi=n32  %s -o - | llvm-readobj -r | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section (3) .rela.text {
// CHECK-NEXT:     0x0 R_MIPS_GPREL16 foo 0x4
// CHECK-NEXT:     0x0 R_MIPS_SUB - 0x0
// CHECK-NEXT:     0x0 R_MIPS_HI16 - 0x0
// CHECK-NEXT:     0x4 R_MIPS_GPREL16 foo 0x4
// CHECK-NEXT:     0x4 R_MIPS_SUB - 0x0
// CHECK-NEXT:     0x4 R_MIPS_LO16 - 0x0
// CHECK-NEXT:   }

  .globl  foo
  .ent  foo
foo:
  lui   $gp, %hi(%neg(%gp_rel(foo+4)))
  addiu $gp, $gp, %lo(%neg(%gp_rel(foo+4)))
  daddu $gp, $gp, $25
  .end  foo
