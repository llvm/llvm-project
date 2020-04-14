declare i32 @foo()

define i32 @bar() {
entry:
  %0 = call i32 @foo()
  ret i32 %0
}

