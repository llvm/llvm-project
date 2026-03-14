; Test that global constructors behind casts are run
;
; RUN: lli -jit-kind=orc %s | FileCheck %s
;
; CHECK: constructor

declare i32 @puts(ptr)

@.str = private constant [12 x i8] c"constructor\00"
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @constructor, ptr null }]

define internal i32 @constructor() #0 {
  %call = tail call i32 @puts(ptr @.str)
  ret i32 0
}

define i32 @main()  {
  ret i32 0
}
