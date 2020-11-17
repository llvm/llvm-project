%a = type { i64 }
%struct = type { i32, i8 }

define void @g(%a* byref(%a)) {
  ret void
}

declare void @baz(%struct* byref(%struct))

define void @foo(%struct* byref(%struct) %a) {
  call void @baz(%struct* byref(%struct) %a)
  ret void
}
