; RUN: llvm-dis < %s.bc | not grep {i32 @llvm\\.punpckh.qdq}
; RUN: llvm-dis < %s.bc | not grep {i32 @llvm\\.punpckl.qdq}
; RUN: llvm-dis < %s.bc | grep shufflevector
