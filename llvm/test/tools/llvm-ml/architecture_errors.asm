; RUN: not llvm-ml -filetype=s %s /Fo - 2>&1 | FileCheck %s
; RUN: not llvm-ml -m32 -filetype=s %s /Fo - 2>&1 | FileCheck %s
; RUN: not llvm-ml64 -m32 -filetype=s %s /Fo - 2>&1 | FileCheck %s

.code

; CHECK: :[[# @LINE + 1]]:5: error: register %rax is only available in 64-bit mode
xor rax, rax

end