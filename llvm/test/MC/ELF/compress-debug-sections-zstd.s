# REQUIRES: zstd

# RUN: llvm-mc -filetype=obj -triple=x86_64 -compress-debug-sections=zstd %s -o %t
# RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SEC
# RUN: llvm-objdump -s %t | FileCheck %s

## Check that the large debug sections .debug_line and .debug_frame are compressed
## and have the SHF_COMPRESSED flag.
# SEC: .nonalloc     PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00     0 0  1
# SEC: .debug_line   PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00   C 0 0  8
# SEC: .debug_abbrev PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00     0 0  1
# SEC: .debug_info   PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00     0 0  1
# SEC: .debug_str    PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 01 MSC 0 0  8
# SEC: .debug_frame  PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00   C 0 0  8

# CHECK:      Contents of section .debug_line
## ch_type == ELFCOMPRESS_ZSTD (2)
# CHECK-NEXT: 0000 02000000 00000000 55010000 00000000
# CHECK-NEXT: 0010 01000000 00000000 {{.*}}

## The compress/decompress round trip should be identical to the uncompressed output.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.uncom
# RUN: llvm-objcopy --decompress-debug-sections %t %t.decom
# RUN: cmp %t.uncom %t.decom

## Don't compress a section not named .debug_*.
	.section .nonalloc,"",@progbits
.rept 50
.asciz "aaaaaaaaaa"
.endr

	.section	.debug_line,"",@progbits

	.section	.debug_abbrev,"",@progbits
.Lsection_abbrev:
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)

	.section	.debug_info,"",@progbits
	.long	12                      # Length of Unit
	.short	4                       # DWARF version number
	.long	.Lsection_abbrev        # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] DW_TAG_compile_unit
	.long	.Linfo_string0          # DW_AT_comp_dir

	.text
foo:
	.cfi_startproc
	.file 1 "Driver.ii"

.rept 3
.ifdef I386
	pushl	%ebp
	.cfi_def_cfa_offset 8
	.cfi_offset %ebp, -8
	movl	%esp, %ebp
	.cfi_def_cfa_register %ebp
	pushl	%ebx
	pushl	%edi
	pushl	%esi
	.cfi_offset %esi, -20
	.cfi_offset %edi, -16
	.cfi_offset %ebx, -12
	.loc	1 1 1 prologue_end
	popl	%esi
	popl	%edi
	popl	%ebx
	popl	%ebp
	.cfi_def_cfa %esp, 4
.else
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.loc	1 1 1 prologue_end
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa %rsp, 8
.endif
.endr

# pad out the line table to make sure it's big enough to warrant compression
	.irpc i, 123456789
	  .irpc j, 0123456789
	    .loc 1 \i\j \j
	    nop
	   .endr
	.endr
	.cfi_endproc
	.cfi_sections .debug_frame

	.section        .debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz  "perfectly compressable data sample *****************************************"
