# RUN: not llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s 2>&1 | FileCheck --strict-whitespace %s

# CHECK: [[#@LINE+2]]:7: error: Expected { at this point
# CHECK: ccmpe 1 rbx, rax
ccmpe 1 rbx, rax

# CHECK: [[#@LINE+2]]:14: error: Expected } or , at this point
# CHECK: ccmpe {sf,cf rbx, rax
ccmpe {sf,cf rbx, rax

# CHECK: [[#@LINE+2]]:8: error: Invalid conditional flags
# CHECK: ccmpe {pf} rbx, rax
ccmpe {pf} rbx, rax

# CHECK: [[#@LINE+2]]:15: error: Duplicated conditional flag
# CHECK: ccmpeq {of,zf,of} rbx, rax
ccmpeq {of,zf,of} rbx, rax

# CHECK: [[#@LINE+2]]:20: error: Expected } at this point
# CHECK: ccmpeq {of,sf,zf,cf,of} rbx, rax
ccmpeq {of,sf,zf,cf,of} rbx, rax
