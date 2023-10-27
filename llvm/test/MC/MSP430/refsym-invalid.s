# RUN: not llvm-mc -triple=msp430 %s 2>&1 | FileCheck %s

# CHECK: [[#@LINE+1]]:8: error: expected identifier in directive
.refsym

# CHECK: [[#@LINE+1]]:9: error: expected identifier in directive
.refsym 42

# CHECK: [[#@LINE+1]]:12: error: expected newline
.refsym sym,
