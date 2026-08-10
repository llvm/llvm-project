%A.11 = type { %B }
%B = type { i8 }
@g1 = external global %A.11

define ptr @use_g1() {
  ret ptr @g1
}
