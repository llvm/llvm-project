; RUN: llvm-as %s -o - | llvm-dis > %t.ll
; RUN: diff %t.ll %s.out

; test 9 bits
;
@b = constant i9 add(i9 511, i9 1)
@c = constant i9 add(i9 511, i9 511)
@d = constant i9 add(i9 504, i9 8)
@e = constant i9 sub(i9 0 , i9 1)
@f = constant i9 sub(i9 0 , i9 511)
@g = constant i9 sub(i9 2 , i9 511)

@h = constant i9 shl(i9 1 , i9 9)
@i = constant i9 shl(i9 1 , i9 8)
@j = constant i9 lshr(i9 511 , i9 8)
@l = constant i9 ashr(i9 511 , i9 8)

@n = constant i9 mul(i9 511, i9 2)
@q = constant i9 sdiv(i9 511, i9 2)
@r = constant i9 udiv(i9 511, i9 2)
@s = constant i9 urem(i9 511, i9 -1)
@t = constant i9 srem(i9 1, i9 510)
@o = constant i9 trunc( i10 512  to i9 )
@p = constant i9 trunc( i10 511  to i9 )

