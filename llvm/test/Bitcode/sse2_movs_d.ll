; RUN: llvm-dis < %s.bc | not grep {i32 @llvm\\.movs.d}
; RUN: llvm-dis < %s.bc | grep shufflevector
