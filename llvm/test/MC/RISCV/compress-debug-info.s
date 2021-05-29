# RUN: llvm-mc -triple riscv32 -mattr=+c -g -M no-aliases %s \
# RUN:   | FileCheck %s -check-prefixes=COMPRESS,BOTH
# RUN: llvm-mc -triple riscv32 -g -M no-aliases %s \
# RUN:   | FileCheck %s -check-prefixes=UNCOMPRESS,BOTH


# This file ensures that compressing an instruction preserves its debug info.


# BOTH-LABEL: .text

# BOTH: .file 1
# BOTH-SAME: "compress-debug-info.s"

# BOTH:            .loc 1 [[# @LINE + 3 ]] 0
# UNCOMPRESS-NEXT: addi a0, a1, 0
# COMPRESS-NEXT:   c.mv a0, a1
addi a0, a1, 0

# BOTH-LABEL: .debug_info
