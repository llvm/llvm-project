; RUN: not llvm-ml -filetype=s %s /Fo - 2>&1 | FileCheck %s --implicit-check-not=error:
; RUN: env INCLUDE=%S not llvm-ml -filetype=s %s /X /Fo - 2>&1 | FileCheck %s --implicit-check-not=error:

; CHECK: :[[# @LINE + 1]]:9: error: Could not find include file 'included.inc'
include included.inc

.code

t1:
mov eax, Const

t2:
; CHECK: :[[# @LINE + 1]]:1: error: invalid instruction mnemonic 'push_pop'
push_pop ebx

end
