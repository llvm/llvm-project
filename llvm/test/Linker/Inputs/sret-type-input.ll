%a = type { i64 }
%struct = type { i32, i8 }

define void @g(ptr sret(%a)) {
  ret void
}

declare void @baz(ptr sret(%struct))

define void @foo(ptr sret(%struct) %a) {
  call void @baz(ptr sret(%struct) %a)
  ret void
}
