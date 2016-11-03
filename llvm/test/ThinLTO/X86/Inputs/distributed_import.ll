@G = internal global i32 7
define i32 @g() {
entry:
  %0 = load i32, i32* @G
  ret i32 %0
}
