# RUN: not llvm-mc -triple=s390x %s 2>&1 | FileCheck %s

# CHECK: [[#@LINE+1]]:9: error: unexpected token in '.machine' directive
.machine

# CHECK: [[#@LINE+1]]:10: error: unexpected token in '.machine' directive
.machine 42

# CHECK: [[#@LINE+1]]:13: error: expected newline
.machine z13+
