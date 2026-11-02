# RUN: split-file %s %t
# RUN: not llvm-mc -triple=amdgpu12.00 -filetype=obj %t/global.s 2>&1 | FileCheck %s --check-prefix=GLOBAL
# RUN: not llvm-mc -triple=amdgpu12.00 -filetype=obj %t/specifier.s 2>&1 | FileCheck %s --check-prefix=SPECIFIER
# RUN: not llvm-mc -triple=amdgpu12.00 -filetype=obj %t/absolute.s 2>&1 | FileCheck %s --check-prefix=ABSOLUTE

# A PC-relative prefetch has a 24-bit field without an ELF relocation. Its
# target must therefore resolve as a local label in the same section.

#--- global.s
.globl global_target
s_prefetch_inst_pc_rel global_target, s10, 7
# GLOBAL: prefetch target must be a local label
global_target:

#--- specifier.s
# Relocation specifiers would select a 32-bit relocation and overwrite the
# adjacent soffset field.
s_prefetch_inst_pc_rel global_target@abs32@lo, s10, 7
# SPECIFIER: an absolute expression or an unqualified label

#--- absolute.s
# A forward-referenced absolute symbol is not a branch/prefetch label.
s_prefetch_inst_pc_rel forward_absolute, s10, 7
.set forward_absolute, 42
# ABSOLUTE: a label
