; RUN: lli -jit-kind=orc-lazy %s

define private void @foo() {
entry:
  ret void
}

define void @"\01l_bar"() {
entry:
  ret void
}

define i32 @main(i32 %argc, ptr nocapture readnone %argv) {
entry:
  call void @foo()
  call void @"\01l_bar"()
  ret i32 0
}
