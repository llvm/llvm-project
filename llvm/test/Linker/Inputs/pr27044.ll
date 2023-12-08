$foo = comdat any
$bar = comdat any

define linkonce_odr i32 @f1() comdat($foo) {
  ret i32 1
}

define void @f2() comdat($foo) {
  call i32 @g2()
  ret void
}

define linkonce_odr i32 @g1() comdat($bar) {
  ret i32 1
}

define linkonce_odr i32 @g2() comdat($bar) {
  ret i32 1
}
