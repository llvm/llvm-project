target triple = "i386-pc-windows"

@.str = private unnamed_addr constant [13 x i8] c"Hello World\0A\00", align 1

define i32 @main() nounwind {
entry:
  %call = tail call i32 @puts(ptr @.str) nounwind
  tail call void @SomeOtherFunction() nounwind
  ret i32 0
}

declare i32 @puts(ptr nocapture) nounwind

declare void @SomeOtherFunction(...)

@var = global i32 0
@llvm.used = appending global [1 x ptr] [ptr @var], section "llvm.metadata"
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr null, ptr null }]
