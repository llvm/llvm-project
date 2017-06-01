%A.11 = type { %B }
%B = type { i8 }
@g1 = external global %A.11

define %A.11* @use_g1() {
  ret %A.11* @g1
}
