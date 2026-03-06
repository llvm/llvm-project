# RUN: not --crash llvm-mc -triple powerpc64-- --filetype=obj < %s 2> %t
# RUN: FileCheck < %t %s
# RUN: not --crash llvm-mc -triple powerpc64le-- --filetype=obj < %s 2> %t
# RUN: FileCheck < %t %s

# CHECK: Unsupported Modifier for fixup_ppc_imm34.
paddi 3, 13, symbol@toc, 0
