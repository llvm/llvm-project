; RUN: not llvm-ml -filetype=s %s /Fo /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

; CHECK: :[[# @LINE + 1]]:8: error: expected name in directive 'extern'
extern 123

; CHECK: :[[# @LINE + 1]]:14: error: expected type in directive 'extern'
extern foo0 :

; CHECK: :[[# @LINE + 1]]:15: error: unrecognized type in directive 'extern'
extern bar0 : typedoesnotexist

extern foo1 : dword, bar1 : word

.code

; CHECK: :[[# @LINE + 1]]:1: error: invalid operand for instruction
mov bx, foo1

; CHECK: :[[# @LINE + 1]]:1: error: invalid operand for instruction
mov bl, bar1

END
