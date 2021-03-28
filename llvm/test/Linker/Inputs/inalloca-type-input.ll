%a = type { i64 }
%struct = type { i32, i8 }

define void @g(%a* inalloca(%a)) {
  ret void
}

declare void @baz(%struct* inalloca(%struct))

define void @foo(%struct* inalloca(%struct) %a) {
  call void @baz(%struct* inalloca(%struct) %a)
  ret void
}
