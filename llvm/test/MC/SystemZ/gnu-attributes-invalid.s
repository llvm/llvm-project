# RUN: not llvm-mc -triple s390x %s 2>&1 | FileCheck %s

# CHECK: [[#@LINE+1]]:1: error: malformed .gnu_attribute directive
.gnu_attribute tag, value

# CHECK: [[#@LINE+1]]:1: error: unrecognized .gnu_attribute tag/value pair.
.gnu_attribute 42, 8

# CHECK: [[#@LINE+1]]:1: error: unrecognized .gnu_attribute tag/value pair.
.gnu_attribute 8, 42

# CHECK: [[#@LINE+1]]:20: error: expected newline
.gnu_attribute 8, 1$
