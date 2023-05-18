; RUN: llvm-as < %s | llvm-dis | grep "quote"
; RUN: verify-uselistorder %s

@"a\22quote" = global i32 0
