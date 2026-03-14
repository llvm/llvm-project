%a = type { i64 }
%struct = type { i32, i8 }

define void @g(ptr inalloca(%a)) {
  ret void
}

declare void @baz(ptr inalloca(%struct))

define void @foo(ptr inalloca(%struct) %a) {
  call void @baz(ptr inalloca(%struct) %a)
  ret void
}
