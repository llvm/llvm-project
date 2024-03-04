# RUN: not llvm-mc -triple x86_64 -show-encoding %s 2>&1 | FileCheck --strict-whitespace %s

# CHECK: error: Expected { at this point
# CHECK: ccmpeq $1 %rax, %rbx
# CHECK:        ^
ccmpeq $1 %rax, %rbx

# CHECK: error: Expected } or , at this point
# CHECK: ccmpeq {sf,cf%rax, %rbx
# CHECK:              ^
ccmpeq {sf,cf%rax, %rbx

# CHECK: error: Invalid conditional flags
# CHECK: ccmpeq {pf} %rax, %rbx
# CHECK:         ^
ccmpeq {pf} %rax, %rbx

# CHECK: error: Duplicated conditional flag
# CHECK: ccmpeq {of,zf,of} %rax, %rbx
# CHECK:               ^
ccmpeq {of,zf,of} %rax, %rbx

# CHECK: error: Expected } at this point
# CHECK: ccmpeq {of,sf,zf,cf,of} %rax, %rbx
# CHECK:                    ^
ccmpeq {of,sf,zf,cf,of} %rax, %rbx
