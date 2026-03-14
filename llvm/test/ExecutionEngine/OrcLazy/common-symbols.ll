; RUN: lli -jit-kind=orc-lazy %s | FileCheck %s
;
; CHECK: 7

@x = common global i32 0, align 4
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 7, ptr @x, align 4
  %0 = load i32, ptr @x, align 4
  %call = call i32 (ptr, ...) @printf(ptr @.str, i32 %0)
  ret i32 0
}

declare i32 @printf(ptr, ...)
