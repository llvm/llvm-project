; RUN: lli -jit-kind=orc-lazy %s
;
; Test handling of global aliases for function and variables.

@x = global i32 42, align 4
@y = alias i32, ptr @x

define i32 @foo() {
entry:
  %0 = load i32, ptr @y, align 4
  ret i32 %0
}

@bar = alias i32(), ptr @foo

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %0 = call i32() @bar()
  %1 = sub i32 %0, 42
  ret i32 %1
}
