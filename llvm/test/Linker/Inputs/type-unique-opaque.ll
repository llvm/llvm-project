%t = type { i8 }
%t2 = type { %t, i16 }

@g = external global %t2
