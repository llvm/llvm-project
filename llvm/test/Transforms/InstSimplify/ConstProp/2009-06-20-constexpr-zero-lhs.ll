; RUN: llvm-as < %s | llvm-dis | not grep ptrtoint
; PR4424
@G = external global i32
@test5 = constant i32 lshr (i32 0, i32 ptrtoint (ptr @G to i32))
@test6 = constant i32 ashr (i32 0, i32 ptrtoint (ptr @G to i32))
@test7 = constant i32 shl (i32 0, i32 ptrtoint (ptr @G to i32))

