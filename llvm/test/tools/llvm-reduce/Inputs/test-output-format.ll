
define void @foo(ptr %ptr) {
  store i32 0, ptr %ptr
  store i32 1, ptr %ptr
  ret void
}
