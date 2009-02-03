; RUN: llvm-as %s -o - | llvm-dis > %t.ll
; RUN: diff %t.ll %s.out

; test 15 bits
;
@b = constant i15 add(i15 32767, i15 1)
@c = constant i15 add(i15 32767, i15 32767)
@d = constant i15 add(i15 32760, i15 8)
@e = constant i15 sub(i15 0 , i15 1)
@f = constant i15 sub(i15 0 , i15 32767)
@g = constant i15 sub(i15 2 , i15 32767)

@h = constant i15 shl(i15 1 , i15 15)
@i = constant i15 shl(i15 1 , i15 14)
@j = constant i15 lshr(i15 32767 , i15 14)
@l = constant i15 ashr(i15 32767 , i15 14)

@n = constant i15 mul(i15 32767, i15 2)
@q = constant i15 mul(i15 -16383,i15 -3)
@r = constant i15 sdiv(i15 -1,   i15 16383)
@s = constant i15 udiv(i15 -1,   i15 16383)
@t = constant i15 srem(i15 1,    i15 32766)
@u = constant i15 urem(i15 32767,i15 -1)
@o = constant i15 trunc( i16 32768  to i15 )
@p = constant i15 trunc( i16 32767  to i15 )
@v = constant i15 srem(i15 -1,    i15 768)
 
