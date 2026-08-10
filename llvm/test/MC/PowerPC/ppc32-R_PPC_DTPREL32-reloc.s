# RUN: llvm-mc -filetype=obj -triple=ppc %s -o %t
# RUN: llvm-readelf -r %t | FileCheck %s 

# CHECK: Relocation section '.rela.debug_info' at offset {{.*}} contains 1 entries:
# CHECK-NEXT: Offset     Info    Type                Sym. Value  Symbol's Name + Addend
# CHECK-NEXT: 00000000  0000014e R_PPC_DTPREL32         00000000   tls_rtp_var + 8000

	.section	.debug_info,"",@progbits
## Used by DW_OP_form_tls_address and DW_OP_GNU_push_tls_address.
	.long	tls_rtp_var@DTPREL+32768

	.section	.tdata,"awT",@progbits
	.globl	tls_rtp_var
tls_rtp_var:
	.long	5
