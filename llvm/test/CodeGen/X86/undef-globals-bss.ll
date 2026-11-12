; RUN: llc < %s | FileCheck %s
target triple = "x86_64-apple-darwin"

; CHECK: .zerofill __DATA,__bss,_vals,8000000,2 ## @vals
@vals = internal unnamed_addr global [2000000 x i32] undef, align 4

; CHECK: .zerofill __DATA,__bss,_struct,8000040,3 ## @struct
@struct = internal global { i1, [8000000 x i8], [7 x i8], i64, { [4 x i32], { i8 }, i1 } }
                { i1 false, [8000000 x i8] zeroinitializer, [7 x i8] undef, i64 0,
                        { [4 x i32], { i8 }, i1 }
                        { [4 x i32] zeroinitializer, { i8 } { i8 undef }, i1 false }
                }, align 8
