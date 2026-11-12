# RUN: not llvm-mc -triple=msp430 %s 2>&1 | FileCheck %s

# CHECK: [[#@LINE+3]]:6: error: unknown token in expression
# CHECK: [[#@LINE+3]]:6: error: unknown token in expression
# CHECK: [[#@LINE+3]]:6: error: unknown token in expression
.byte, 42
.word, 42
.long, 42

# CHECK: [[#@LINE+3]]:10: error: unknown token in expression
# CHECK: [[#@LINE+3]]:10: error: unknown token in expression
# CHECK: [[#@LINE+3]]:10: error: unknown token in expression
.byte 42,
.word 42,
.long 42,

# CHECK: [[#@LINE+3]]:10: error: unexpected token
# CHECK: [[#@LINE+3]]:10: error: unexpected token
# CHECK: [[#@LINE+3]]:10: error: unexpected token
.byte 42 42
.word 42 42
.long 42 42
