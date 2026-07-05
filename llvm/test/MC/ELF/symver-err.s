# RUN: not llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

# CHECK: {{.*}}.s:[[#@LINE+2]]:1: error: multiple versions for multi
.symver multi, multi@1
.symver multi, multi@2

.symver equiv, equiv@1
.symver equiv, equiv@1

# CHECK: {{.*}}.s:[[#@LINE+1]]:1: error: default version symbol undefined@@v1 must be defined
.symver undefined_2, undefined@@v1
.long undefined_2
