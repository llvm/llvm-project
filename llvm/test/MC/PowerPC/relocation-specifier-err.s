# RUN: not llvm-mc -triple powerpc64 --filetype=obj %s -o %t 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple powerpc64le --filetype=obj %s -o %t 2>&1 | FileCheck %s

# CHECK: [[#@LINE+1]]:4: error: unsupported relocation type
bl foo@toc

# CHECK: [[#@LINE+1]]:12: error: unsupported relocation type
addi 3, 3, foo@plt

# CHECK: [[#@LINE+1]]:14: error: unsupported relocation type
paddi 3, 13, foo@toc, 0

# CHECK: [[#@LINE+1]]:15: error: unsupported relocation type
ld %r5, p_end - .(%r5)

# CHECK: [[#@LINE+1]]:7: error: unsupported relocation type
.quad foo@toc
