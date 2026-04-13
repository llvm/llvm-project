; RUN: llvm-ml64 -filetype=s %s /Fo - | FileCheck %s --implicit-check-not=error:
; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s --implicit-check-not=error:

.code

xor rax, rax
; CHECK: xor rax, rax

end
