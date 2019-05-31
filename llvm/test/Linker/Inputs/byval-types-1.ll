%struct = type {i32, i8}

declare void @baz(%struct* byval(%struct))

define void @foo(%struct* byval(%struct) %a) {
  call void @baz(%struct* byval(%struct) %a)
  ret void
}
