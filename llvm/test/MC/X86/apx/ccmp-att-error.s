# RUN: not llvm-mc -triple x86_64 -show-encoding %s 2>&1 | FileCheck --strict-whitespace %s

# CHECK: [[#@LINE+2]]:8: error: Expected { at this point
# CHECK: ccmpeq $1 %rax, %rbx
ccmpeq $1 %rax, %rbx

# CHECK: [[#@LINE+2]]:14: error: Expected } or , at this point
# CHECK: ccmpeq {sf,cf%rax, %rbx
ccmpeq {sf,cf%rax, %rbx

# CHECK: [[#@LINE+2]]:9: error: Invalid conditional flags
# CHECK: ccmpeq {pf} %rax, %rbx
ccmpeq {pf} %rax, %rbx

# CHECK: [[#@LINE+2]]:15: error: Duplicated conditional flag
# CHECK: ccmpeq {of,zf,of} %rax, %rbx
ccmpeq {of,zf,of} %rax, %rbx

# CHECK: [[#@LINE+2]]:20: error: Expected } at this point
# CHECK: ccmpeq {of,sf,zf,cf,of} %rax, %rbx
ccmpeq {of,sf,zf,cf,of} %rax, %rbx
