// RUN: llvm-mc -triple aarch64-elf -filetype asm %s | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -triple aarch64-elf -filetype obj %s \
// RUN:   | llvm-readelf -s - | FileCheck %s --check-prefix=OBJ

// ASM:      .variant_pcs local
// ASM-NEXT: local:
.text
.variant_pcs local
local:

/// Binding directive before .variant_pcs.
// ASM:      .globl def1
// ASM-NEXT: .variant_pcs def1
// ASM-NEXT: def1:
.global def1
.variant_pcs def1
def1:

/// .variant_pcs before binding directive.
// ASM:      .variant_pcs def2
// ASM-NEXT: .weak def2
// ASM-NEXT: def2:
.variant_pcs def2
.weak def2
def2:

.globl alias_def1
.set alias_def1, def1

// ASM:      .variant_pcs undef
.variant_pcs undef

// OBJ:      NOTYPE LOCAL  DEFAULT [VARIANT_PCS]  [[#]] local
// OBJ-NEXT: NOTYPE GLOBAL DEFAULT [VARIANT_PCS]  [[#]] def1
// OBJ-NEXT: NOTYPE WEAK   DEFAULT [VARIANT_PCS]  [[#]] def2
// OBJ-NEXT: NOTYPE GLOBAL DEFAULT                [[#]] alias_def1
// OBJ-NEXT: NOTYPE GLOBAL DEFAULT [VARIANT_PCS]  UND   undef
