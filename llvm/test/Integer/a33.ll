; RUN: llvm-as %s -o - | llvm-dis > %t.ll
; RUN: diff %t.ll %s.out

; test 33 bits
;
@b = constant i33 add(i33 8589934591, i33 1)
@c = constant i33 add(i33 8589934591, i33 8589934591)
@d = constant i33 add(i33 8589934584, i33 8)
@e = constant i33 sub(i33 0 , i33 1)
@f = constant i33 sub(i33 0 , i33 8589934591)
@g = constant i33 sub(i33 2 , i33 8589934591)

@h = constant i33 shl(i33 1 , i33 33)
@i = constant i33 shl(i33 1 , i33 32)
@j = constant i33 lshr(i33 8589934591 , i33 32)
@l = constant i33 ashr(i33 8589934591 , i33 32)

@n = constant i33 mul(i33 8589934591, i33 2)
@q = constant i33 sdiv(i33 -1,        i33 4294967295)
@r = constant i33 udiv(i33 -1,        i33 4294967295)
@s = constant i33 srem(i33  1,        i33 8589934590)
@t = constant i33 urem(i33 8589934591,i33 -1)
@o = constant i33 trunc( i34 8589934592 to i33 )
@p = constant i33 trunc( i34 8589934591  to i33 )
@u = constant i33 srem(i33  -1,       i33 17)
 
