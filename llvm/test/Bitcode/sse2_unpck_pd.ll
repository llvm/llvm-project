; RUN: llvm-dis < %s.bc | not grep {i32 @llvm\\.unpckh.pd}
; RUN: llvm-dis < %s.bc | not grep {i32 @llvm\\.unpckl.pd}
; RUN: llvm-dis < %s.bc | grep shufflevector
