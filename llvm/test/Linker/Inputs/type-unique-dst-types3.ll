%A.11 = type opaque
@g2 = external global %A.11

define ptr @use_g2() {
  ret ptr @g2
}
