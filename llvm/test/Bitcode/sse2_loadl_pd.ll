; RUN: llvm-dis < %s.bc | not grep {i32 @llvm\\.loadl.pd}
; RUN: llvm-dis < %s.bc | grep shufflevector
