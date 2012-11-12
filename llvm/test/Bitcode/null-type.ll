; RUN: not llvm-dis < %s.bc > /dev/null 2> %t
; RUN: FileCheck %s < %t
; PR8494

; CHECK: Invalid MODULE_CODE_FUNCTION record
