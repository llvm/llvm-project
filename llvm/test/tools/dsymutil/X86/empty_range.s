# This test verifies that an empty range list in the .debug_ranges section
# doesn't crash llvm-dsymutil. As clang does not produce this kind of debug
# info anymore, we used this hand-crafted assembly file to produce a testcase
# Compile with:
#        llvm-mc -triple x86_64-apple-darwin -filetype=obj -o 1.o empty_range.o

# RUN: llvm-dsymutil -f -y %p/dummy-debug-map.map -oso-prepend-path %p/../Inputs/empty_range -o - | llvm-dwarfdump -debug-dump=info - | FileCheck %s

        .section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 11
	.globl	__Z3foov
	.align	4, 0x90
__Z3foov:                               ## @_Z3foov
Lfunc_begin0:
	pushq	%rbp
	movq	%rsp, %rbp
	popq	%rbp
	retq
Lfunc_end0:
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                       ## Abbreviation Code
	.byte	17                      ## DW_TAG_compile_unit
	.byte	1                       ## DW_CHILDREN_yes
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	2                       ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	0                       ## DW_CHILDREN_no
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
        .byte   0x55                    ## DW_AT_ranges
        .byte   6                       ## DW_FORM_data4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	0                       ## EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
	.long	22                      ## Length of Unit
	.short	2                       ## DWARF version number
	.long	0                       ## Offset Into Abbrev. Section
	.byte	8                       ## Address Size (in bytes)
	.byte	1                       ## Abbrev [1]  DW_TAG_compile_unit
	.byte	2                       ## Abbrev [2] DW_TAG_subprogram
	.quad	Lfunc_begin0            ## DW_AT_low_pc
        .long   0                       ## DW_AT_ranges (pointing at an empty entry)
	.byte	0                       ## End Of Children Mark
	.section	__DWARF,__debug_ranges,regular,debug
Ldebug_range:
        .long 0
        .long 0

# CHECK:  DW_TAG_compile_unit
# CHECK:    DW_TAG_subprogram
# CHECK-NEXT:      DW_AT_low_pc{{.*}}(0x0000000000010000)
# CHECK-NEXT:      DW_AT_ranges{{.*}}(0x00000000)

# There was a bug that would use the currently active object file when a
# debug map object isn't found. Check that we only linked one file.
# CHECK-NOT: DW_TAG_compile_unit

