# RUN: not llvm-mc -triple=wasm32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

foo:
.section .foo,"S",@

# CHECK: {{.*}}.s:[[# @LINE+1]]:1: error: changed section flags for .foo, expected: 0x1
.section .foo,"",@
