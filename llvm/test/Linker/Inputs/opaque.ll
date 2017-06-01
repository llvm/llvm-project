%A = type { }
%B = type { %D, %E, %B* }

%D = type { %E }
%E = type opaque

@g2 = external global %A
@g3 = external global %B

define void @f1()  {
  getelementptr %A, %A* null, i32 0
  ret void
}

define %A* @use_g2() {
 ret %A* @g2
}

define %B* @use_g3() {
  ret %B* @g3
}
