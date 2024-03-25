define i32 @g() {
entry:
  ret i32 0
}

$f = comdat nodeduplicate

define i32 @f() comdat {
entry:
  %0 = call i32 @g()
  ret i32 %0
}
