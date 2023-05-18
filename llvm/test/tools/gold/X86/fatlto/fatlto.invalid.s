# REQUIRES: x86_64-linux

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: not %gold -plugin %llvmshlibdir/LLVMgold%shlibext %t -o /dev/null 2>&1 | FileCheck %s

# CHECK: error:{{.*}} Invalid bitcode signature

.section        .llvm.lto,"e",@progbits
.Lllvm.embedded.object:
        .asciz  "BC\300\3365\000"
        .size   .Lllvm.embedded.object, 12
