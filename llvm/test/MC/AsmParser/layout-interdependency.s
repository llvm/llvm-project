# RUN: not llvm-mc --filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

fct_end:

# CHECK: layout-interdependency.s:[[#@LINE+1]]:7: error: expected assembly-time absolute expression
.fill (data_start - fct_end), 1, 42
# CHECK: layout-interdependency.s:[[#@LINE+1]]:7: error: expected assembly-time absolute expression
.fill (fct_end - data_start), 1, 42

data_start:
