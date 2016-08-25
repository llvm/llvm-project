target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  call void (...) @globalfunc()
  ret i32 0
}

declare void @globalfunc(...)
