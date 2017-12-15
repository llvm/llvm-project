; RUN: not %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    %p/Inputs/invalid.bc -o %t2 2>&1 | FileCheck %s

; test that only one error gets printed

; CHECK: error: LLVM gold plugin has failed to create LTO module: Malformed block
; CHECK-NOT: error
