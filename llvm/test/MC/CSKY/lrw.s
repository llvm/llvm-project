# RUN: llvm-mc -filetype=obj -triple=csky -mattr=+e2 < %s \
# RUN:     | llvm-objdump --mattr=+e2  --no-show-raw-insn -M no-aliases -d -r - | FileCheck %s


.data
sec:
    .long 0x77
.text
tstart:
    lrw r0,lnk
    lrw r0,lnk - 4
    lrw r0,lnk + 4
    .short 0x1C00
    lrw r0,sec
    lrw r0,sec - 4
    lrw r0,sec + 4
    lrw r0,0
    lrw r0,0xFFFF
    lrw r31,0
.L1:
    lrw r31,.L1
.L2:
    lrw r0, .L2
.L3:
    lrw r0, .L3 - 64*1024
.L4:
    lrw r0, .L4 + 64*1024 - 2

    lrw r0,0x01020304
    lrw r0,0xFFFFFFFE

# CHECK:        0:      	lrw16	r0, 0x28 <$d.4>
# CHECK-NEXT:   2:      	lrw16	r0, 0x2c  <$d.4+0x4>
# CHECK-NEXT:   4:      	lrw16	r0, 0x30  <$d.4+0x8>

# CHECK:        6:	        00 1c	.short	0x1c00

# CHECK:        8:      	lrw16	r0, 0x34 <$d.4+0xc>
# CHECK-NEXT:   a:      	lrw16	r0, 0x38  <$d.4+0x10>
# CHECK-NEXT:   c:      	lrw16	r0, 0x3c <$d.4+0x14>
# CHECK-NEXT:   e:      	movi16	r0, 0
# CHECK-NEXT:  10:      	movi32	r0, 65535
# CHECK-NEXT:  14:      	movi32	r31, 0
# CHECK-NEXT:  18:              lrw32	r31, 0x40 <$d.4+0x18>
# CHECK-NEXT:  1c:              lrw16	r0,  0x44 <$d.4+0x1c>
# CHECK-NEXT:  1e:              lrw16	r0,  0x48 <$d.4+0x20>
# CHECK-NEXT:  20:              lrw16	r0,  0x4c <$d.4+0x24>
# CHECK-NEXT:  22:              lrw16	r0,  0x50 <$d.4+0x28>
# CHECK-NEXT:  24:              lrw16	r0,  0x54 <$d.4+0x2c>

# CHECK:       28:	    00 00 00 00	.word	0x00000000
# CHECK-NEXT:       	      00000028:  R_CKCORE_ADDR32	lnk
# CHECK-NEXT:  2c:	    00 00 00 00	.word	0x00000000
# CHECK-NEXT:       	      0000002c:  R_CKCORE_ADDR32	lnk-0x4
# CHECK-NEXT:  30:	    00 00 00 00	.word	0x00000000
# CHECK-NEXT:       	      00000030:  R_CKCORE_ADDR32	lnk+0x4
# CHECK-NEXT:  34:	    00 00 00 00	.word	0x00000000
# CHECK-NEXT:       	      00000034:  R_CKCORE_ADDR32	.data
# CHECK-NEXT:  38:	    00 00 00 00	.word	0x00000000
# CHECK-NEXT:       	      00000038:  R_CKCORE_ADDR32	.data-0x4
# CHECK-NEXT:  3c:	    00 00 00 00	.word	0x00000000
# CHECK-NEXT:       	      0000003c:  R_CKCORE_ADDR32	.data+0x4
# CHECK-NEXT:  40:	    00 00 00 00	.word	0x00000000
# CHECK-NEXT:       	     00000040:  R_CKCORE_ADDR32	.text+0x18
# CHECK-NEXT:  44:	    00 00 00 00	.word	0x00000000
# CHECK-NEXT:       	     00000044:  R_CKCORE_ADDR32	.text+0x1c
# CHECK-NEXT:  48:	    00 00 00 00	.word	0x00000000
# CHECK-NEXT:       	     00000048:  R_CKCORE_ADDR32	.text-0xffe2
# CHECK-NEXT:  4c:	    00 00 00 00	.word	0x00000000
# CHECK-NEXT:       	     0000004c:  R_CKCORE_ADDR32	.text+0x1001e
# CHECK-NEXT:  50:	    04 03 02 01	.word	0x01020304
# CHECK-NEXT:  54:	    fe ff ff ff	.word	0xfffffffe

