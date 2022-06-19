; Test that global constructors behind casts are run
;
; RUN: lli -jit-kind=orc %s | FileCheck %s
;
; CHECK: constructor

declare i32 @puts(i8*)

@.str = private constant [12 x i8] c"constructor\00"
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* bitcast (i32 ()* @constructor to void ()*), i8* null }]

define internal i32 @constructor() #0 {
  %call = tail call i32 @puts(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i64 0, i64 0))
  ret i32 0
}

define i32 @main()  {
  ret i32 0
}
