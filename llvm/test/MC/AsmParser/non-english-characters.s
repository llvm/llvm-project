# RUN: llvm-mc -triple i386-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-readobj %t | FileCheck %s
# CHECK: Format: ELF32-i386

# 0b—
# 0x—
# .—4
# .X—
# .1—
# .1e—
# 0x.—
# 0x0p—
.intel_syntax
# 1—
