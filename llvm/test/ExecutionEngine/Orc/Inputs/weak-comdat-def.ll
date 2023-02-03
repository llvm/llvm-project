$c = comdat any

define i32 @f() comdat($c) {
entry:
  ret i32 0
}
