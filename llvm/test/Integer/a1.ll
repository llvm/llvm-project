; RUN: llvm-as %s -o - | llvm-dis > %t.ll
; RUN: diff %t.ll %s.out

; test 1 bit
;
@b = constant i1 add(i1 1 , i1 1)
@c = constant i1 add(i1 -1, i1 1)
@d = constant i1 add(i1 -1, i1 -1)
@e = constant i1 sub(i1 -1, i1 1)
@f = constant i1 sub(i1 1 , i1 -1)
@g = constant i1 sub(i1 1 , i1 1)

@h = constant i1 shl(i1 1 , i1 1)  ; undefined
@i = constant i1 shl(i1 1 , i1 0)
@j = constant i1 lshr(i1 1, i1 1)  ; undefined
@m = constant i1 ashr(i1 1, i1 1)  ; undefined

@n = constant i1 mul(i1 -1, i1 1)
@o = constant i1 sdiv(i1 -1, i1 1) ; overflow
@p = constant i1 sdiv(i1 1 , i1 -1); overflow
@q = constant i1 udiv(i1 -1, i1 1)
@r = constant i1 udiv(i1 1, i1 -1)
@s = constant i1 srem(i1 -1, i1 1) ; overflow
@t = constant i1 urem(i1 -1, i1 1)
@u = constant i1 srem(i1  1, i1 -1) ; overflow
