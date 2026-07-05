; RUN: not llvm-ml -filetype=s %s /Fo /dev/null 2>&1 | FileCheck %s

; CHECK: :[[# @LINE+1]]:1: error: expected section directive
foo PROC
; CHECK: :[[# @LINE+1]]:6: error: expected section directive before assembly directive
  ret
foo ENDP
