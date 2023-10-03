# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-pc-win32 -filetype=obj -o %t/COFF_x86_64.o %s
# RUN: llvm-rtdyld -triple=x86_64-pc-win32  -dummy-extern external_func=0x1 \
# RUN:   -dummy-extern external_data=0x2 -verify -check=%s %t/COFF_x86_64.o


	.section section,"rx"
section:
	.long 0
Lreloc:
	.long 0
# rtdyld-check: *{2}Lreloc = 1
	.reloc 4, secidx, section+4


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
# %bb.0:                                # %entry
.Ltmp1:
	.seh_endprologue
# rtdyld-check: decode_operand(inst1, 4) = __real400921f9f01b866e - next_pc(inst1)
inst1:
	movsd	__real400921f9f01b866e(%rip), %xmm0 # xmm0 = mem[0],zero
	retq
.Leh_func_end0:
.Ltmp2:
	.seh_endproc

	.globl	call_to_dllimport
        .p2align	4, 0x90
# Check calls to dllimports.
#
# (1) Check that callq argument points to synthesized stub addr.
# rtdyld-check: decode_operand(call_to_dllimport, 3) = \
# rtdyld-check:   stub_addr(COFF_x86_64.o/.text, __imp_external_func) - \
# rtdyld-check:     next_pc(call_to_dllimport)
#
# (2) Check that synthesized stub entry points to call target.
# rtdyld-check: *{8}(stub_addr(COFF_x86_64.o/.text, __imp_external_func)) = \
# rtdyld-check:   external_func
call_to_dllimport:
	callq	*__imp_external_func(%rip)

        .globl  load_from_dllimport
        .p2align        4, 0x90
# Check loads from dllimports.
#
# (1) Check that the movq argument points to synthesized stub addr.
# rtdyld-check: decode_operand(load_from_dllimport, 4) = \
# rtdyld-check:   stub_addr(COFF_x86_64.o/.text, __imp_external_data) - \
# rtdyld-check:     next_pc(load_from_dllimport)
#
# (2) Check that synthesized stub entry points to datao target.
# rtdyld-check: *{8}(stub_addr(COFF_x86_64.o/.text, __imp_external_data)) = \
# rtdyld-check:   external_data
load_from_dllimport:
	movq	__imp_external_data(%rip), %rax

        .data
	.globl  x                       # @x
# rtdyld-check: *{8}x = F
x:
	.quad	F

# Make sure the JIT doesn't bail out on BSS sections.
        .bss
bss_check:
        .fill 8, 1, 0
