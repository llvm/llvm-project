; RUN: not llvm-dis < %s.bc 2>&1 | FileCheck %s
; PR8494

; CHECK: Malformed block
