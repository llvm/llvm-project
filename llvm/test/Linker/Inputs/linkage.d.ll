@Y = global i8 42

define i64 @foo() { ret i64 7 }

@llvm.used = appending global [2 x ptr] [ptr @Y, ptr @foo], section "llvm.metadata"
