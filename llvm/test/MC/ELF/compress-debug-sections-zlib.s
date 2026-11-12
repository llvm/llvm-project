// REQUIRES: zlib

// Check zlib style
// RUN: llvm-mc -filetype=obj -compress-debug-sections=zlib -triple x86_64-pc-linux-gnu < %s -o %t
// RUN: llvm-objdump -s %t | FileCheck %s
// RUN: llvm-dwarfdump -debug-str %t | FileCheck --check-prefix=STR %s
// RUN: llvm-readelf --sections %t | FileCheck --check-prefixes=FLAGS,FLAGS64 %s

// RUN: llvm-mc -filetype=obj -compress-debug-sections=zlib -triple i386-pc-linux-gnu --defsym I386=1 %s -o %t
// RUN: llvm-readelf -S -s %t | FileCheck --check-prefixes=386-SYMBOLS,FLAGS,FLAGS32 %s

// Decompress one valid dwarf section just to check that this roundtrips,
// we use .debug_str section for that
// STR: perfectly compressable data sample *****************************************


// Now check the zlib style output:

// Don't compress small sections, such as this simple debug_abbrev example
// CHECK: Contents of section .debug_abbrev:
// CHECK-NOT: ZLIB
// CHECK-NOT: Contents of
// CHECK: Contents of section .debug_info:
// FIXME: Handle compressing alignment fragments to support compressing debug_frame
// CHECK: Contents of section .debug_frame:
// CHECK-NOT: ZLIB
// CHECK: Contents of

# FLAGS:   .text         PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00 AX 0 0  4
# FLAGS:   .nonalloc     PROGBITS [[#%x,]] [[#%x,]] 000226   00    0 0  1

## Check that the large .debug_line and .debug_frame have the SHF_COMPRESSED
## flag.
# FLAGS32: .debug_line   PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00  C 0 0  4
# FLAGS32: .debug_abbrev PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00    0 0  1
# FLAGS32: .debug_info   PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00    0 0  1
# FLAGS32: .debug_frame  PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00  C 0 0  4

# FLAGS64: .debug_line   PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00  C 0 0  8
# FLAGS64: .debug_abbrev PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00    0 0  1
# FLAGS64: .debug_info   PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00    0 0  1
# FLAGS64: .debug_frame  PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 00  C 0 0  8

# 386-SYMBOLS: Symbol table '.symtab'
# 386-SYMBOLS: .debug_str

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

# Below is the section we will use to check that after compression with llvm-mc,
# llvm-dwarfdump tool will be able to decompress data back and dump it. Data sample
# should be compressable enough, so it is filled with some amount of equal symbols at the end
	.section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "perfectly compressable data sample *****************************************"
