# RUN: not llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s 2>&1 | FileCheck --strict-whitespace %s

# CHECK: [[#@LINE+2]]:7: error: Expected { at this point
# CHECK: ccmpe 1 rbx, rax
ccmpe 1 rbx, rax

# CHECK: [[#@LINE+2]]:8: error: Expected dfv at this point
# CHECK: ccmpe {sf} rbx, rax
ccmpe {sf} rbx, rax

# CHECK: [[#@LINE+2]]:11: error: Expected = at this point
# CHECK: ccmpe {dfv:sf} rbx, rax
ccmpe {dfv:sf} rbx, rax

# CHECK: [[#@LINE+2]]:18: error: Expected } or , at this point
# CHECK: ccmpe {dfv=sf,cf rbx, rax
ccmpe {dfv=sf,cf rbx, rax

# CHECK: [[#@LINE+2]]:12: error: Invalid conditional flags
# CHECK: ccmpe {dfv=pf} rbx, rax
ccmpe {dfv=pf} rbx, rax

# CHECK: [[#@LINE+2]]:19: error: Duplicated conditional flag
# CHECK: ccmpeq {dfv=of,zf,of} rbx, rax
ccmpeq {dfv=of,zf,of} rbx, rax

# CHECK: [[#@LINE+2]]:24: error: Expected } at this point
# CHECK: ccmpeq {dfv=of,sf,zf,cf,of} rbx, rax
ccmpeq {dfv=of,sf,zf,cf,of} rbx, rax
