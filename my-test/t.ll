; t.ll
target triple = "x86_64-unknown-linux-gnu"  ; 根据您的架构调整

define i32 @f(i32 %a, i32 %b, i32 %c) {
entry:
  %x = add i32 %a, %b
  %y = add i32 %x, %c
  ret i32 %y
}
