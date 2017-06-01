define linkonce_odr i32 @baz() #0 {
entry:
  ret i32 0
}

define i8* @bar() {
entry:
  ret i8* bitcast (i32 ()* @baz to i8*)
}
