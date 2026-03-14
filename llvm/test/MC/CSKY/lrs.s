# RUN: llvm-mc -filetype=obj -triple=csky -mattr=+2e3  < %s \
# RUN:     | llvm-objdump --mattr=+2e3 --no-show-raw-insn -M no-aliases -d -r - | FileCheck %s


.data
sec:
    .long 0x77
.text
tstart:
    lrs32.b r0,[lnk]
    lrs32.b r0,[lnk - 4]
    lrs32.b r0,[lnk + 4]
    .short 0x1C00
    lrs32.h r0,[sec]
    lrs32.h r0,[sec - 4]
    lrs32.h r0,[sec + 4]
    lrs32.b r0,[0]
    lrs32.b r0,[0xFFFF]
    lrs32.b r31,[0]
.L1:
    lrs32.w r31,[.L1]
.L2:
    lrs32.w r0, [.L2]
.L3:
    lrs32.w r0, [.L3 - 64*1024]
.L4:
    lrs32.w r0, [.L4 + 64*1024 - 2]

# CHECK:        0:      	lrs32.b	r0, [0]
# CHECK-NEXT:   			        00000000:  R_CKCORE_DOFFSET_IMM18	lnk
# CHECK-NEXT:   4:      	lrs32.b	r0, [0]
# CHECK-NEXT:   			        00000004:  R_CKCORE_DOFFSET_IMM18	lnk-0x4
# CHECK-NEXT:   8:      	lrs32.b	r0, [0]
# CHECK-NEXT:   			        00000008:  R_CKCORE_DOFFSET_IMM18	lnk+0x4

# CHECK:        c:	        00 1c		.short	0x1c00

# CHECK:        e:       lrs32.h	r0, [0]
# CHECK-NEXT:   			        0000000e:  R_CKCORE_DOFFSET_IMM18_2	.data
# CHECK-NEXT:   12:      lrs32.h	r0, [0]
# CHECK-NEXT:   			        00000012:  R_CKCORE_DOFFSET_IMM18_2	.data-0x4
# CHECK-NEXT:   16:      lrs32.h	r0, [0]
# CHECK-NEXT:   			        00000016:  R_CKCORE_DOFFSET_IMM18_2	.data+0x4
# CHECK-NEXT:   1a:      lrs32.b	r0, [0]
# CHECK-NEXT:   			        0000001a:  R_CKCORE_DOFFSET_IMM18	*ABS*
# CHECK-NEXT:   1e:      lrs32.b	r0, [0]
# CHECK-NEXT:   			        0000001e:  R_CKCORE_DOFFSET_IMM18	*ABS*+0xffff
# CHECK-NEXT:   22:      lrs32.b	r31, [0]
# CHECK-NEXT:   			        00000022:  R_CKCORE_DOFFSET_IMM18	*ABS*
# CHECK-NEXT:   26:      lrs32.w	r31, [0]
# CHECK-NEXT:   			        00000026:  R_CKCORE_DOFFSET_IMM18_4	.text+0x26
# CHECK-NEXT:   2a:      lrs32.w	r0, [0]
# CHECK-NEXT:   			        0000002a:  R_CKCORE_DOFFSET_IMM18_4	.text+0x2a
# CHECK-NEXT:   2e:      lrs32.w	r0, [0]
# CHECK-NEXT:   			        0000002e:  R_CKCORE_DOFFSET_IMM18_4	.text-0xffd2
# CHECK-NEXT:   32:      lrs32.w	r0, [0]
# CHECK-NEXT:   			        00000032:  R_CKCORE_DOFFSET_IMM18_4	.text+0x10030
