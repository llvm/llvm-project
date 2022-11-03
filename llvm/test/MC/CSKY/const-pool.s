# RUN: llvm-mc -filetype=obj -triple=csky -mattr=+e2 -mattr=+high-registers < %s \
# RUN:     | llvm-objdump --mattr=+e2  --no-show-raw-insn -M no-aliases -d -r - | FileCheck %s


lrw a3, [.LCPI1_1]
.zero 0x3ea
.LCPI1_1:
    .long   symA@GOTOFF

lrw a3, [.LCPI1_2]
.zero 0x3ec
.LCPI1_2:
    .long   symA@GOTOFF


# CHECK:        0:      	lrw16	r3, 0x3ec
# CHECK:      3f0:              lrw32	r3, 0x7e0

