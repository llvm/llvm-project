%a = type { i64 }
%struct = type { i32, i8 }

define void @g(%a* sret(%a)) {
  ret void
}

declare void @baz(%struct* sret(%struct))

define void @foo(%struct* sret(%struct) %a) {
  call void @baz(%struct* sret(%struct) %a)
  ret void
}
