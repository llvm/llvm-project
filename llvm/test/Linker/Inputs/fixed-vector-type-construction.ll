%t = type {i32, float}
define void @foo(<4 x %t*> %x) {
  ret void
}
