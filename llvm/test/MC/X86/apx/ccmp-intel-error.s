# RUN: not llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s 2>&1 | FileCheck --strict-whitespace %s

# CHECK: error: Expected { at this point
# CHECK: ccmpe 1 rbx, rax
# CHECK:       ^
ccmpe 1 rbx, rax

# CHECK: error: Expected } or , at this point
# CHECK: ccmpe {sf,cf rbx, rax
# CHECK:              ^
ccmpe {sf,cf rbx, rax

# CHECK: error: Invalid conditional flags
# CHECK: ccmpe {pf} rbx, rax
# CHECK:        ^
ccmpe {pf} rbx, rax

# CHECK: error: Duplicated conditional flag
# CHECK: ccmpeq {of,zf,of} rbx, rax
# CHECK:               ^
ccmpeq {of,zf,of} rbx, rax

# CHECK: error: Expected } at this point
# CHECK: ccmpeq {of,sf,zf,cf,of} rbx, rax
# CHECK:                    ^
ccmpeq {of,sf,zf,cf,of} rbx, rax
