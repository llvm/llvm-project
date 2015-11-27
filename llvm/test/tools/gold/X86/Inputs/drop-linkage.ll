$foo = comdat any
define linkonce void @foo() comdat {
  ret void
}

define void @bar() {
  call void @foo()
  ret void
}
