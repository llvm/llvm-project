; RUN: llvm-as %s -o - | llvm-dis > %t.ll
; RUN: diff %t.ll %s.out

; test 63 bits
;
@b = constant i63 add(i63 9223372036854775807, i63 1)
@c = constant i63 add(i63 9223372036854775807, i63 9223372036854775807)
@d = constant i63 add(i63 9223372036854775800, i63 8)
@e = constant i63 sub(i63 0 , i63 1)
@f = constant i63 sub(i63 0 , i63 9223372036854775807)
@g = constant i63 sub(i63 2 , i63 9223372036854775807)

@h = constant i63 shl(i63 1 , i63 63)
@i = constant i63 shl(i63 1 , i63 62)
@j = constant i63 lshr(i63 9223372036854775807 , i63 62)
@l = constant i63 ashr(i63 9223372036854775807 , i63 62)

@n = constant i63 mul(i63 9223372036854775807, i63 2) 
@q = constant i63 sdiv(i63 -1,                 i63 4611686018427387903)
@u = constant i63 sdiv(i63 -1,                 i63 1)
@r = constant i63 udiv(i63 -1,                 i63 4611686018427387903)
@s = constant i63 srem(i63  3,                 i63 9223372036854775806)
@t = constant i63 urem(i63 9223372036854775807,i63 -1)
@o = constant i63 trunc( i64 9223372036854775808 to i63 )
@p = constant i63 trunc( i64 9223372036854775807  to i63 )
