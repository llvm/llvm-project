; RUN: not llvm-ml -filetype=s %s /Fo /dev/null 2>&1 | FileCheck %s

.code

foo PROC
  ret
foo ENDP

bar PROC
  ret
bar ENDP

t1:
alias foo = bar
alias foo = <bar>
alias <foo> = bar

; CHECK: error: expected <aliasName>
; CHECK: error: expected <aliasName>
; CHECK: error: expected <actualName>

t2:
alias <foo> <bar>
alias <foo>, <bar>

; CHECK: error: unexpected token in alias directive
; CHECK: error: unexpected token in alias directive

t3:
alias <foo = bar>
alias <foo = bar

; CHECK: error: unexpected token in alias directive
; CHECK: error: expected <aliasName>

END