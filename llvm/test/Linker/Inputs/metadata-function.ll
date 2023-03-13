define weak void @foo() !weak !0 {
  unreachable
}

define void @baz() !baz !0 {
  unreachable
}

define void @b() !b !0 {
  unreachable
}

!0 = !{!"b"}
