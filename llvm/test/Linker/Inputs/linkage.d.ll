@Y = global i8 42

define i64 @foo() { ret i64 7 }

@llvm.used = appending global [2 x i8*] [i8* @Y, i8* bitcast (i64 ()* @foo to i8*)], section "llvm.metadata"
