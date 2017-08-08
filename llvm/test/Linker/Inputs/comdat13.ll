$foo = comdat any
@foo = internal global i8 1, comdat
define i8* @zed() {
  call void @bax()
  ret i8* @foo
}
define internal void @bax() comdat($foo) {
  ret void
}
