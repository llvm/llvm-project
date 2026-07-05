; RUN: not llvm-ml -filetype=s %s /Fo - 2>&1 | FileCheck %s --implicit-check-not=error:

.code

t1 PROC
  xor eax, eax
t1_nested PROC
  ret
t1 ENDP
t1_nested ENDP
; CHECK: :[[# @LINE - 2]]:1: error: endp does not match current procedure 't1_nested'

END
