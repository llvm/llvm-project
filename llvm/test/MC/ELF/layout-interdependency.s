# RUN: not llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s

fct_end:

# CHECK: layout-interdependency.s:[[#@LINE+1]]:7: error: invalid number of bytes
.fill (data_start - fct_end), 1, 42
# CHECK: layout-interdependency.s:[[#@LINE+1]]:7: error: invalid number of bytes
.fill (fct_end - data_start), 1, 42

data_start:
