# RUN: not llvm-mc -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s

# CHECK: {{.*}}.s:6:1: error: unmatched .ifs or .elses
.if 1
.else
