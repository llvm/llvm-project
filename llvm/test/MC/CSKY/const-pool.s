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

# CHECK: 000007e0:  R_CKCORE_GOTOFF	symA
# CHECK: 000007e4:  R_CKCORE_GOT32	va1
# CHECK: 000007e8:  R_CKCORE_GOTOFF	va2
# CHECK: 000007ec:  R_CKCORE_PLT32	va3
# CHECK: 000007f0:  R_CKCORE_TLS_GD32	va4
# CHECK: 000007f4:  R_CKCORE_TLS_LDM32	va5
# CHECK: 000007f8:  R_CKCORE_TLS_LE32	va6


.LCPI0_0:
    .long   va1@GOT
    .long   va2@GOTOFF
    .long   va3@PLT
    .long   va4@TLSGD
    .long   va5@TLSLDM
    .long   va6@TPOFF
