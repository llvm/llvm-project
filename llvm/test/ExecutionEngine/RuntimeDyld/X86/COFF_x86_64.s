# RUN: llvm-mc -triple=x86_64-pc-win32 -filetype=obj -o %T/COFF_x86_64.o %s
# RUN: llvm-rtdyld -triple=x86_64-pc-win32 -verify -check=%s %/T/COFF_x86_64.o
        	.text
	.def	 F;
	.scl	2;
	.type	32;
	.endef
	.globl	__real400921f9f01b866e
	.section	.rdata,"dr",discard,__real400921f9f01b866e
	.align	8
__real400921f9f01b866e:
	.quad	4614256650576692846     # double 3.1415899999999999
	.text
	.globl	F
        .global inst1
	.align	16, 0x90
F:                                      # @F
.Ltmp0:
.seh_proc F
# BB#0:                                 # %entry
.Ltmp1:
	.seh_endprologue
# rtdyld-check: decode_operand(inst1, 4) = __real400921f9f01b866e - next_pc(inst1)
inst1:
	movsd	__real400921f9f01b866e(%rip), %xmm0 # xmm0 = mem[0],zero
	retq
.Leh_func_end0:
.Ltmp2:
	.seh_endproc

        .data
	.globl  x                       # @x
# rtdyld-check: *{8}x = F
x:
	.quad	F

# Make sure the JIT doesn't bail out on BSS sections.
        .bss
bss_check:
        .fill 8, 1, 0
