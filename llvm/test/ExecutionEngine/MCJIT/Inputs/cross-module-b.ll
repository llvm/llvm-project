declare i32 @FA()

define i32 @FB() {
  %r = call i32 @FA( )   ; <i32> [#uses=1]
  ret i32 %r
}

