@used1 = global i8 4
@used2 = global i32 123

@llvm.used = appending global [2 x i8*] [
   i8* @used1,
   i8* bitcast (i32* @used2 to i8*)
], section "llvm.metadata"
