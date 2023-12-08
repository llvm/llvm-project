%A = type { }
%B = type { %D, %E, ptr }

%D = type { %E }
%E = type opaque

@g2 = external global %A
@g3 = external global %B

define void @f1()  {
  getelementptr %A, ptr null, i32 0
  ret void
}

define ptr @use_g2() {
 ret ptr @g2
}

define ptr @use_g3() {
  ret ptr @g3
}
