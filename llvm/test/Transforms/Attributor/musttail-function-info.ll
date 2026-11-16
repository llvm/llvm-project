; RUN: opt -passes=attributor -disable-output %s

; Initializing @caller's information adds @callee's, which grows the map.

define void @f0() {
  ret void
}

define void @f1() {
  ret void
}

define void @f2() {
  ret void
}

define void @f3() {
  ret void
}

define void @f4() {
  ret void
}

define void @f5() {
  ret void
}

define void @f6() {
  ret void
}

define void @f7() {
  ret void
}

define void @f8() {
  ret void
}

define void @f9() {
  ret void
}

define void @f10() {
  ret void
}

define void @f11() {
  ret void
}

define void @f12() {
  ret void
}

define void @f13() {
  ret void
}

define void @f14() {
  ret void
}

define void @f15() {
  ret void
}

define void @f16() {
  ret void
}

define void @f17() {
  ret void
}

define void @f18() {
  ret void
}

define void @f19() {
  ret void
}

define void @f20() {
  ret void
}

define void @f21() {
  ret void
}

define void @f22() {
  ret void
}

define void @f23() {
  ret void
}

define void @f24() {
  ret void
}

define void @f25() {
  ret void
}

define void @f26() {
  ret void
}

define void @f27() {
  ret void
}

define void @f28() {
  ret void
}

define void @f29() {
  ret void
}

define void @f30() {
  ret void
}

define void @f31() {
  ret void
}

define void @f32() {
  ret void
}

define void @f33() {
  ret void
}

define void @f34() {
  ret void
}

define void @f35() {
  ret void
}

define void @f36() {
  ret void
}

define void @f37() {
  ret void
}

define void @f38() {
  ret void
}

define void @f39() {
  ret void
}

define void @f40() {
  ret void
}

define void @f41() {
  ret void
}

define void @f42() {
  ret void
}

define void @f43() {
  ret void
}

define void @f44() {
  ret void
}

define void @f45() {
  ret void
}

define void @caller() {
  musttail call void @callee()
  ret void
}

define void @callee() {
  ret void
}
