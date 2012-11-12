; RUN: llvm-as < %s | llvm-dis | grep "quote"

@"a\22quote" = global i32 0
