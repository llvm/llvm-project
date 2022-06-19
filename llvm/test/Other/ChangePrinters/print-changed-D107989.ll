; D107989 This triggered an assert

; RUN: opt  -passes=globalopt < %s -disable-output -print-changed=diff-quiet

define signext i32 @main()  {
entry:
  ret i32 0
}

define internal void @f() {
entry:
  ret void
}
