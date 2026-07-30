define weak void @foo() !weak !0 {
  unreachable
}

define void @baz() !baz !0 {
  unreachable
}

define void @b() !b !0 {
  unreachable
}

%AltHandle = type { i8* }
declare !types !1 %AltHandle @init.AltHandle()

define void @uses.AltHandle() {
  %.res = call %AltHandle @init.AltHandle()
  unreachable
}

!0 = !{!"b"}
!1 = !{%AltHandle undef}
