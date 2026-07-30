define weak i32 @baz() #0 {
entry:
  ret i32 0
}

define ptr @bar() {
entry:
  ret ptr @baz
}
