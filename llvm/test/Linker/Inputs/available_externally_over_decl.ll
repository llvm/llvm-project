@h = global ptr @f
@h2 = global ptr @g

define available_externally void @f() {
  ret void
}

define available_externally void @g() {
  ret void
}
