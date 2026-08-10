%struct = type {i32, i8}

declare void @baz(ptr byval(%struct))

define void @foo(ptr byval(%struct) %a) {
  call void @baz(ptr byval(%struct) %a)
  ret void
}
