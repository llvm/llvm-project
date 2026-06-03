define void @store_i64(ptr %p, i64 %v) {
  store i64 %v, ptr %p, align 8
  ret void
}
