; RUN: llvm-dis < %s.bc | not grep {i32 @llvm\\.shuf.pd}
; RUN: llvm-dis < %s.bc | grep shufflevector
