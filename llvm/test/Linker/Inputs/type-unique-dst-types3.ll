%A.11 = type opaque
@g2 = external global %A.11

define %A.11* @use_g2() {
  ret %A.11* @g2
}
