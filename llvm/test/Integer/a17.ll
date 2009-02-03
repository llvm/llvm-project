; RUN: llvm-as %s -o - | llvm-dis > %t.ll
; RUN: diff %t.ll %s.out

; test 17 bits
;
@b = constant i17 add(i17 131071, i17 1)
@c = constant i17 add(i17 131071, i17 131071)
@d = constant i17 add(i17 131064, i17 8)
@e = constant i17 sub(i17 0 , i17 1)
@f = constant i17 sub(i17 0 , i17 131071)
@g = constant i17 sub(i17 2 , i17 131071)

@h = constant i17 shl(i17 1 , i17 17)
@i = constant i17 shl(i17 1 , i17 16)
@j = constant i17 lshr(i17 131071 , i17 16)
@l = constant i17 ashr(i17 131071 , i17 16)

@n = constant i17 mul(i17 131071, i17 2) 
@q = constant i17 sdiv(i17 -1,    i17 65535)
@r = constant i17 udiv(i17 -1,    i17 65535)
@s = constant i17 srem(i17  1,    i17 131070)
@t = constant i17 urem(i17 131071,i17 -1)
@o = constant i17 trunc( i18 131072  to i17 )
@p = constant i17 trunc( i18 131071  to i17 )
@v = constant i17 srem(i17  -1,    i17 15)
