# RUN: not llvm-mc -filetype=obj -triple=i386 %s -o /dev/null 2>&1 | FileCheck %s

# CHECK: :[[#@LINE+1]]:7: error: unsupported relocation type
.quad foo
# CHECK: :[[#@LINE+1]]:8: error: unsupported relocation type
.8byte foo
