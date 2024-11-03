%t = type { i8 }
%t2 = type { %t, i16 }

define %t2 @f() {
  ret %t2 { %t { i8 0 }, i16 0 }
}
