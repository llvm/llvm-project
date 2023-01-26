// RUN: llvm-mc -triple riscv64 -filetype obj -o - %s | llvm-readobj --symbols - | FileCheck %s
// RUN: llvm-mc -triple riscv64 -filetype obj -defsym=OBJ=1 -o - %s | llvm-readelf -s - | FileCheck %s --check-prefix=OBJ
// RUN: not llvm-mc -triple riscv64 -filetype asm -defsym=ERR=1 -o - %s 2>&1 | FileCheck %s --check-prefix=ERR

.text
.variant_cc local
local:

// CHECK: Name: local
// CHECK: Other [ (0x80)

.ifdef OBJ
/// Binding directive before .variant_cc.
.global def1
.variant_cc def1
def1:

/// Binding directive after .variant_cc.
.variant_cc def2
.weak def2
def2:

.globl alias_def1
.set alias_def1, def1

.variant_cc undef

// OBJ:      NOTYPE LOCAL  DEFAULT [VARIANT_CC] [[#]] local
// OBJ-NEXT: NOTYPE GLOBAL DEFAULT [VARIANT_CC] [[#]] def1
// OBJ-NEXT: NOTYPE WEAK   DEFAULT [VARIANT_CC] [[#]] def2
// OBJ-NEXT: NOTYPE GLOBAL DEFAULT              [[#]] alias_def1
// OBJ-NEXT: NOTYPE GLOBAL DEFAULT [VARIANT_CC] UND   undef
.endif

.ifdef ERR
.variant_cc
// ERR: [[#@LINE-1]]:12: error: expected symbol name

.global fox
.variant_cc fox bar
// ERR: [[#@LINE-1]]:17: error: expected newline
.endif
