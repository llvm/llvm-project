add.d $a0, $a1, $a2
addi.d $a0, $a1, 1
and $a0, $a1, $a2
nor $a0, $a1, $a2
slt $a0, $a1, $a2
maskeqz $a0, $a1, $a2
lu12i.w $a0, 1
sll.d $a0, $a1, $a2
slli.d $a0, $a1, 1
rotr.d $a0, $a1, $a2
alsl.d $a0, $a1, $a2, 1
bytepick.d $a0, $a1, $a2, 1
bstrins.d $a0, $a1, 15, 8
bstrpick.d $a0, $a1, 15, 8
ext.w.b $a0, $a1
clo.d $a0, $a1
ctz.d $a0, $a1
revb.d $a0, $a1
bitrev.d $a0, $a1
pcaddi $a0, 1
b 4
beq $a0, $a1, 8
