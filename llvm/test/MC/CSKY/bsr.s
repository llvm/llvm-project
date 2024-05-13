# RUN: llvm-mc -filetype=obj -triple=csky  < %s \
# RUN:     | llvm-objdump  --no-show-raw-insn -M no-aliases -d -r - | FileCheck %s

.data
sec:
    .long 0x77
.text
tstart:
    bsr lnk
    bsr lnk - 4
    bsr lnk + 4
    .short 0x1C00
    bsr sec
    bsr sec - 4
    bsr sec + 4

.L1:
    bsr .L1
.L2:
    bsr .L2 - 1024
.L3:
    bsr .L3 + 1022

.L4:
    bsr .L4 - 1026
.L5:
    bsr .L5 + 1024

.L6:
    bsr .L6 - 64*1024*1024
.L7:
    bsr .L7 + 64*1024*1024 - 2


# CHECK:       0:      	bsr32	0x0
# CHECK:       			        00000000:  R_CKCORE_PCREL_IMM26_2	lnk
# CHECK:       4:      	bsr32	0x4
# CHECK:       			        00000004:  R_CKCORE_PCREL_IMM26_2	lnk-0x4
# CHECK:       8:      	bsr32	0x8
# CHECK:       			        00000008:  R_CKCORE_PCREL_IMM26_2	lnk+0x4

# CHECK:       e:      	bsr32	0xe
# CHECK:       			        0000000e:  R_CKCORE_PCREL_IMM26_2	.data
# CHECK:       12:      bsr32	0x12
# CHECK:       			        00000012:  R_CKCORE_PCREL_IMM26_2	.data-0x4
# CHECK:       16:      bsr32	0x16
# CHECK:       			        00000016:  R_CKCORE_PCREL_IMM26_2	.data+0x4
# CHECK:       1a:      bsr32	0x1a
# CHECK:       1e:      bsr32	0xfffffc1e
# CHECK:       22:      bsr32	0x420
# CHECK:       26:      bsr32	0xfffffc24
# CHECK:       2a:      bsr32	0x42a
# CHECK:       2e:      bsr32	0xfc00002e
# CHECK:       32:      bsr32	0x4000030