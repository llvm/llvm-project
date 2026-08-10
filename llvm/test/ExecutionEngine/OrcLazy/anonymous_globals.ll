; RUN: lli -jit-kind=orc-lazy %s

define private void @0() {
entry:
  ret void
}

define private void @"\01L_foo"() {
entry:
  ret void
}

define i32 @main(i32 %argc, ptr nocapture readnone %argv) {
entry:
  call void @0()
  tail call void @"\01L_foo"()
  ret i32 0
}
