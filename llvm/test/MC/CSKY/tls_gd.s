# RUN: llvm-mc -filetype=obj -triple=csky -mattr=+2e3  < %s \
# RUN:     | llvm-objdump  --mattr=+2e3 --no-show-raw-insn -M no-aliases -d -r - | FileCheck %s

.LTLS0:
	lrw16 r0, xxx@TLSGD32
	grs32 r2, .LTLS0
	addu16 r0, r0, r2
	lrw16 r3, __tls_get_addr@PLT
	ldr32.w r3, (rgb, r3 << 0)
	jsr16 r3

# CHECK:            0:      	lrw16	r0, 0x10
# CHECK-NEXT:       2:      	grs32	r2, 0x0
# CHECK-NEXT:       6:      	addu16	r0, r0, r2
# CHECK-NEXT:       8:      	lrw16	r3, 0x14
# CHECK-NEXT:       a:      	ldr32.w	r3, (r28, r3 << 0)
# CHECK-NEXT:       e:      	jsr16	r3

# CHECK:           10:	00 00 00 00	.word	0x00000000
# CHECK-NEXT:           00000010:  R_CKCORE_TLS_GD32	xxx+0x10
# CHECK-NEXT:      14:	00 00 00 00	.word	0x00000000
# CHECK-NEXT:           00000014:  R_CKCORE_PLT32	__tls_get_addr
