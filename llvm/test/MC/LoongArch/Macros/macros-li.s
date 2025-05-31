# RUN: llvm-mc --triple=loongarch64 %s | FileCheck %s

li.w $a0, 0x0
# CHECK:      ori $a0, $zero, 0
li.d $a0, 0x0
# CHECK-NEXT: ori $a0, $zero, 0

li.w $a0, 0xfff
# CHECK:      ori $a0, $zero, 4095
li.d $a0, 0xfff
# CHECK-NEXT: ori $a0, $zero, 4095

li.w $a0, 0x7ffff000
# CHECK:      lu12i.w $a0, 524287
li.d $a0, 0x7ffff000
# CHECK-NEXT:  lu12i.w $a0, 524287

li.w $a0, 0x80000000
# CHECK:      lu12i.w $a0, -524288
li.d $a0, 0x80000000
# CHECK-NEXT: lu12i.w $a0, -524288
# CHECK-NEXT: lu32i.d $a0, 0

li.w $a0, 0xfffff800
# CHECK:      addi.w $a0, $zero, -2048
li.d $a0, 0xfffff800
# CHECK-NEXT: addi.w $a0, $zero, -2048
# CHECK-NEXT: lu32i.d $a0, 0

li.w $a0, 0xfffffffffffff800
# CHECK:      addi.w $a0, $zero, -2048
li.d $a0, 0xfffffffffffff800
# CHECK-NEXT: addi.w $a0, $zero, -2048

li.w $a0, 0xffffffff80000800
# CHECK:      lu12i.w $a0, -524288
# CHECK-NEXT: ori $a0, $a0, 2048
li.d $a0, 0xffffffff80000800
# CHECK-NEXT: lu12i.w $a0, -524288
# CHECK-NEXT: ori $a0, $a0, 2048

li.d $a0, 0x7ffff00000800
# CHECK:      ori $a0, $zero, 2048
# CHECK-NEXT: lu32i.d $a0, 524287

li.d $a0, 0x8000000000fff
# CHECK:      ori $a0, $zero, 4095
# CHECK-NEXT: bstrins.d $a0, $a0, 51, 51

li.d $a0, 0x8000080000800
# CHECK:      lu12i.w $a0, -524288
# CHECK-NEXT: ori $a0, $a0, 2048
# CHECK-NEXT: lu32i.d $a0, -524288
# CHECK-NEXT: lu52i.d $a0, $a0, 0

li.d $a0, 0x80000fffff800
# CHECK:      addi.w $a0, $zero, -2048
# CHECK-NEXT: lu32i.d $a0, -524288
# CHECK-NEXT: lu52i.d $a0, $a0, 0

li.d $a0, 0xffffffffff000
# CHECK:      lu12i.w $a0, -1
# CHECK-NEXT: lu52i.d $a0, $a0, 0

li.d $a0, 0xffffffffff800
# CHECK:      addi.w $a0, $zero, -2048
# CHECK-NEXT: lu52i.d $a0, $a0, 0

li.d $a0, 0x7ff0000000000000
# CHECK:      lu52i.d $a0, $zero, 2047

li.d $a0, 0x7ff0000080000000
# CHECK:      lu12i.w $a0, -524288
# CHECK-NEXT: lu32i.d $a0, 0
# CHECK-NEXT: lu52i.d $a0, $a0, 2047

li.d $a0, 0x7fffffff800007ff
# CHECK:      lu12i.w $a0, -524288
# CHECK-NEXT: ori $a0, $a0, 2047
# CHECK-NEXT: lu52i.d $a0, $a0, 2047

li.d $a0, 0xfff0000000000fff
# CHECK:      ori $a0, $zero, 4095
# CHECK-NEXT: lu52i.d $a0, $a0, -1

li.d $a0, 0xffffffff7ffff800
# CHECK:      lu12i.w $a0, 524287
# CHECK-NEXT: ori $a0, $a0, 2048
# CHECK-NEXT: lu32i.d $a0, -1
