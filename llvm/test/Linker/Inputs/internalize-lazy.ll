define linkonce_odr void @g() {
  ret void
}

define void @f() {
  call void @g()
  ret void
}
