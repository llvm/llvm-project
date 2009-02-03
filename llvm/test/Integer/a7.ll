; RUN: llvm-as %s -o - | llvm-dis > %t.ll
; RUN: diff %t.ll %s.out

; test 7 bits
;
@b = constant i7 add(i7 127, i7 1)
@q = constant i7 add(i7 -64, i7 -1)
@c = constant i7 add(i7 127, i7 127)
@d = constant i7 add(i7 120, i7 8)
@e = constant i7 sub(i7 0 , i7 1)
@f = constant i7 sub(i7 0 , i7 127)
@g = constant i7 sub(i7 2 , i7 127)
@r = constant i7 sub(i7 -3, i7 120)
@s = constant i7 sub(i7 -3, i7 -8)

@h = constant i7 shl(i7 1 , i7 7)
@i = constant i7 shl(i7 1 , i7 6)
@j = constant i7 lshr(i7 127 , i7 6)
@l = constant i7 ashr(i7 127 , i7 6)
@m2= constant i7 ashr(i7 -1  , i7 3)

@n = constant i7 mul(i7 127, i7 2)
@t = constant i7 mul(i7 -63, i7 -2)
@u = constant i7 mul(i7 -32, i7 2)
@v = constant i7 sdiv(i7 -1, i7 63)
@w = constant i7 udiv(i7 -1, i7 63)
@x = constant i7 srem(i7 1 , i7 126)
@y = constant i7 urem(i7 127, i7 -1)
@o = constant i7 trunc( i8 128  to i7 )
@p = constant i7 trunc( i8 255  to i7 )
 
