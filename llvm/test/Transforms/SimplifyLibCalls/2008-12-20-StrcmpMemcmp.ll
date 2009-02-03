; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | grep call.*memcmp

@.str = internal constant [2 x i8] c"x\00"

declare i32 @strcmp(i8* %dest, i8* %src)

define i32 @foo(i8* %x, i8* %y) {
  %A = call i32 @strcmp(i8* %x, i8* getelementptr ([2 x i8]* @.str, i32 0, i32 0))
  ret i32 %A
}
