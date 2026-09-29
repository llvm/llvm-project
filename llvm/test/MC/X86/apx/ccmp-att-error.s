# RUN: not llvm-mc -triple x86_64 -show-encoding %s 2>&1 | FileCheck --strict-whitespace %s

# CHECK: [[#@LINE+2]]:8: error: Expected { at this point
# CHECK: ccmpeq $1 %rax, %rbx
ccmpeq $1 %rax, %rbx

# CHECK: [[#@LINE+2]]:9: error: Expected dfv at this point
# CHECK: ccmpeq {sf} %rax, %rbx
ccmpeq {sf} %rax, %rbx

# CHECK: [[#@LINE+2]]:12: error: Expected = at this point
# CHECK: ccmpeq {dfv:sf} %rax, %rbx
ccmpeq {dfv:sf} %rax, %rbx

# CHECK: [[#@LINE+2]]:18: error: Expected } or , at this point
# CHECK: ccmpeq {dfv=sf,cf%rax, %rbx
ccmpeq {dfv=sf,cf%rax, %rbx

# CHECK: [[#@LINE+2]]:13: error: Invalid conditional flags
# CHECK: ccmpeq {dfv=pf} %rax, %rbx
ccmpeq {dfv=pf} %rax, %rbx

# CHECK: [[#@LINE+2]]:19: error: Duplicated conditional flag
# CHECK: ccmpeq {dfv=of,zf,of} %rax, %rbx
ccmpeq {dfv=of,zf,of} %rax, %rbx

# CHECK: [[#@LINE+2]]:24: error: Expected } at this point
# CHECK: ccmpeq {dfv=of,sf,zf,cf,of} %rax, %rbx
ccmpeq {dfv=of,sf,zf,cf,of} %rax, %rbx
