target triple = "x86_64-unknown-linux-gnu"
declare i32 @g()
define i32 @main() {
  call i32 @g()
  ret i32 0
}
