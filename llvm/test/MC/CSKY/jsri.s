# RUN: llvm-mc -filetype=obj -triple=csky -mattr=+2e3  < %s \
# RUN:     | llvm-objdump --mattr=+2e3 --no-show-raw-insn -M no-aliases -d -r - | FileCheck %s


.data
sec:
    .long 0x77
.text
tstart:
    jsri lnk
    jsri lnk - 4
    jsri lnk + 4
    .short 0x1C00
    jsri sec
    jsri sec - 4
    jsri sec + 4

.J1:
    jsri .J1
.J2:
    jsri .J2 - 0x1000
.J3:
    jsri .J3 + 0x1000

    jsri 0x01020304
    jsri 0xFFFFFFFE


# CHECK:         0:      	jsri32  0x30 <$d.4>
# CHECK-NEXT:    4:      	jsri32	0x34 <$d.4+0x4>
# CHECK-NEXT:    8:      	jsri32	0x38 <$d.4+0x8>

# CHECK:         c:	        00 1c	.short	0x1c00

# CHECK:         e:      	jsri32	0x3c <$d.4+0xc>
# CHECK-NEXT:   12:         jsri32	0x40 <$d.4+0x10>
# CHECK-NEXT:   16:         jsri32	0x44 <$d.4+0x14>



# CHECK:    <.J1>:
# CHECK-NEXT:   1a:         bsr32	0x1a

# CHECK:    <.J2>:
# CHECK-NEXT:   1e:         bsr32	0xfffff01e <$d.4+0xffffffffffffefee>

# CHECK:    <.J3>:
# CHECK-NEXT:   22:         bsr32	0x1022 <$d.4+0xff2>
# CHECK-NEXT:   26:         jsri32	0x54 <$d.4+0x24>
# CHECK-NEXT:   2a:         jsri32	0x58 <$d.4+0x28>
# CHECK-NEXT:   2e:         bkpt


# CHECK:        30:      00 00 00 00     .word	0x0
# CHECK-NEXT:   			        00000030:  R_CKCORE_ADDR32	lnk
# CHECK-NEXT:   34:      00 00 00 00     .word	0x0
# CHECK-NEXT:   			        00000034:  R_CKCORE_ADDR32	lnk-0x4
# CHECK-NEXT:   38:      00 00 00 00     .word	0x0
# CHECK-NEXT:   			        00000038:  R_CKCORE_ADDR32	lnk+0x4
# CHECK-NEXT:   3c:      00 00 00 00     .word	0x0
# CHECK-NEXT:   			        0000003c:  R_CKCORE_ADDR32	.data
# CHECK-NEXT:   40:      00 00 00 00     .word	0x0
# CHECK-NEXT:   			        00000040:  R_CKCORE_ADDR32	.data-0x4
# CHECK-NEXT:   44:      00 00 00 00     .word	0x0
# CHECK-NEXT:   			        00000044:  R_CKCORE_ADDR32	.data+0x4
# CHECK-NEXT:   48:      00 00 00 00     .word	0x0
# CHECK-NEXT:   			        00000048:  R_CKCORE_ADDR32	.text+0x1a
# CHECK-NEXT:   4c:      00 00 00 00     .word	0x0
# CHECK-NEXT:   			        0000004c:  R_CKCORE_ADDR32	.text-0xfe2
# CHECK-NEXT:   50:      00 00 00 00     .word	0x0
# CHECK-NEXT:   			        00000050:  R_CKCORE_ADDR32	.text+0x1022
# CHECK-NEXT:   54:      04 03 02 01	 .word	0x01020304
# CHECK-NEXT:   58:      fe ff ff ff     .word	0xfffffffe
