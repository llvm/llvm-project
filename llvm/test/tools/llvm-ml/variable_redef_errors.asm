; RUN: not llvm-ml -filetype=s %s /Fo - 2>&1 | FileCheck %s --implicit-check-not=error:

.data

; <var> EQU <expression> can't be redefined to a new value
equated_number equ 3
; CHECK: :[[# @LINE + 1]]:21: error: invalid variable redefinition
equated_number equ 4

; CHECK: :[[# @LINE + 1]]:1: error: cannot redefine a built-in symbol
@Line equ 5

; CHECK: :[[# @LINE + 1]]:1: error: cannot redefine a built-in symbol
@Version equ 6

.code

end
