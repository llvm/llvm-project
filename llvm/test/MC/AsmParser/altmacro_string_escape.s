# RUN: llvm-mc -triple i386-linux-gnu %s| FileCheck %s

.altmacro
# single-character string escape
# To include any single character literally in a string
# (even if the character would otherwise have some special meaning),
# you can prefix the character with `!'.
# For example, you can write `<4.3 !> 5.4!!>' to get the literal text `4.3 > 5.4!'.

# CHECK: workForFun:
.macro fun1 number
  .if \number=5
    lableNotWork:
  .else
    workForFun:
  .endif
.endm

# CHECK: workForFun2:
.macro fun2 string
  .if \string
    workForFun2:
  .else
    notworkForFun2:
  .endif
.endm

fun1 <5!!>
fun2 <5!>4>
