; RUN: llvm-dis < %s.bc | not grep {i32 @llvm\\.movl.dq}
; RUN: llvm-dis < %s.bc | grep shufflevector
