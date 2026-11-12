# RUN: not llvm-mc -triple x86_64 %s -o /dev/null 2>&1 | FileCheck %s

# CHECK: {{.*}}.s:[[# @LINE+1]]:30: error: expected linked-to symbol
.section .link,"ao",@progbits

# CHECK: {{.*}}.s:[[# @LINE+1]]:31: error: invalid linked-to symbol
.section .link,"ao",@progbits,123

# CHECK: {{.*}}.s:[[# @LINE+1]]:31: error: linked-to symbol is not in a section: foo
.section .link,"ao",@progbits,foo

# CHECK: {{.*}}.s:[[# @LINE+2]]:31: error: linked-to symbol is not in a section: bar
bar = 42
.section .link,"ao",@progbits,bar

# CHECK: {{.*}}.s:[[# @LINE+2]]:31: error: linked-to symbol is not in a section: baz
.quad baz
.section .link,"ao",@progbits,baz
