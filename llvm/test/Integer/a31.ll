; RUN: llvm-as %s -o - | llvm-dis > %t.ll
; RUN: diff %t.ll %s.out

; test 31 bits
;
@b = constant i31 add(i31 2147483647, i31 1)
@c = constant i31 add(i31 2147483647, i31 2147483647)
@d = constant i31 add(i31 2147483640, i31 8)
@e = constant i31 sub(i31 0 , i31 1)
@f = constant i31 sub(i31 0 , i31 2147483647)
@g = constant i31 sub(i31 2 , i31 2147483647)

@h = constant i31 shl(i31 1 , i31 31)
@i = constant i31 shl(i31 1 , i31 30)
@j = constant i31 lshr(i31 2147483647 , i31 30)
@l = constant i31 ashr(i31 2147483647 , i31 30)

@n = constant i31 mul(i31 2147483647, i31 2)
@q = constant i31 sdiv(i31 -1,        i31 1073741823)
@r = constant i31 udiv(i31 -1,        i31 1073741823)
@s = constant i31 srem(i31  1,        i31 2147483646)
@t = constant i31 urem(i31 2147483647,i31 -1)
@o = constant i31 trunc( i32 2147483648  to i31 )
@p = constant i31 trunc( i32 2147483647  to i31 ) 
@u = constant i31 srem(i31 -3,        i31 17)
