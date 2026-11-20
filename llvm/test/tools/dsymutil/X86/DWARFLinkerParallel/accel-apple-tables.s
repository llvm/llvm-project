# Check the Apple accelerator tables of the parallel linker: names whose hashes
# collide are ordered by name within their hash, the values of a name are
# ordered by DIE offset, and the type table has its extra atoms.

# RUN: llvm-mc -triple x86_64-apple-darwin -filetype=obj %s -o %t.o
# RUN: echo '---' > %t.map
# RUN: echo "triple:          'x86_64-apple-darwin'" >> %t.map
# RUN: echo 'objects:'  >> %t.map
# RUN: echo " -  filename: '%t.o'" >> %t.map
# RUN: echo '    symbols:' >> %t.map
# RUN: echo '      - { sym: __Z2ezi, objAddr: 0x0, binAddr: 0x10000, size: 0x20 }' >> %t.map
# RUN: echo '      - { sym: __ZN1N2FYENS_5OuterE, objAddr: 0x20, binAddr: 0x10020, size: 0x10 }' >> %t.map
# RUN: echo '      - { sym: __ZN1N2EzENS_5OuterE, objAddr: 0x30, binAddr: 0x10030, size: 0x10 }' >> %t.map
# RUN: echo '      - { sym: __Z2Ezi, objAddr: 0x40, binAddr: 0x10040, size: 0x10 }' >> %t.map
# RUN: echo '      - { sym: _main, objAddr: 0x50, binAddr: 0x10050, size: 0x70 }' >> %t.map
# RUN: echo '...' >> %t.map

# RUN: dsymutil --linker=parallel --accelerator=Apple -y %t.map -f -o %t.dSYM
# RUN: llvm-dwarfdump --verify %t.dSYM | FileCheck %s --check-prefix=VERIFY
# RUN: llvm-dwarfdump --apple-names --apple-types %t.dSYM | FileCheck %s

## The output does not depend on the number of threads.
# RUN: dsymutil --linker=parallel --accelerator=Apple -y %t.map -f -o %t.2.dSYM --num-threads 2
# RUN: llvm-dwarfdump --apple-names --apple-types %t.dSYM | sed 1d > %t.txt
# RUN: llvm-dwarfdump --apple-names --apple-types %t.2.dSYM | sed 1d > %t.2.txt
# RUN: diff %t.txt %t.2.txt

# VERIFY: No errors.

## "Ez" and "FY" have the same hash, and so do their mangled names. FY() is
## defined first, but the names are ordered by name within their hash.
# CHECK:      .apple_names contents:
# CHECK:        Hash 0x5973a4 [
# CHECK-NEXT:     Name@{{.*}} {
# CHECK-NEXT:       String: {{.*}} "Ez"
# CHECK-NEXT:       Data 0 [
# CHECK-NEXT:         Atom[0]: [[N_EZ:0x[0-9a-f]+]]
# CHECK-NEXT:       ]
# CHECK-NEXT:       Data 1 [
# CHECK-NEXT:         Atom[0]: [[EZ:0x[0-9a-f]+]]
# CHECK-NEXT:       ]
# CHECK-NEXT:     }
# CHECK-NEXT:     Name@{{.*}} {
# CHECK-NEXT:       String: {{.*}} "FY"
# CHECK-NEXT:       Data 0 [
# CHECK-NEXT:         Atom[0]: [[N_FY:0x[0-9a-f]+]]
# CHECK-NEXT:       ]
# CHECK-NEXT:     }
# CHECK-NEXT:   ]
# CHECK:          String: {{.*}} "_ZN1N2EzENS_5OuterE"
# CHECK-NEXT:       Data 0 [
# CHECK-NEXT:         Atom[0]: [[N_EZ]]
# CHECK-NEXT:       ]
# CHECK-NEXT:     }
# CHECK-NEXT:     Name@{{.*}} {
# CHECK-NEXT:       String: {{.*}} "_ZN1N2FYENS_5OuterE"
# CHECK-NEXT:       Data 0 [
# CHECK-NEXT:         Atom[0]: [[N_FY]]
# CHECK-NEXT:       ]
# CHECK-NEXT:     }
# CHECK-NEXT:   ]

# CHECK:      .apple_types contents:
# CHECK:          String: {{.*}} "int"
# CHECK-NEXT:       Data 0 [
# CHECK-NEXT:         Atom[0]: {{.*}}
# CHECK-NEXT:         Atom[1]: 0x0024 (DW_TAG_base_type)
# CHECK-NEXT:         Atom[2]: 0x00
# CHECK-NEXT:         Atom[3]: {{.*}}
# CHECK-NEXT:       ]
# CHECK-NEXT:       Data 1 [
# CHECK-NEXT:         Atom[0]: {{.*}}
# CHECK-NEXT:         Atom[1]: 0x0024 (DW_TAG_base_type)

## The "Inner" of ez() is recorded before N::Outer::Inner, but comes after it
## in the type unit.
# CHECK:          String: {{.*}} "Inner"
# CHECK-NEXT:       Data 0 [
# CHECK-NEXT:         Atom[0]: 0x0000003f
# CHECK-NEXT:         Atom[1]: 0x0013 (DW_TAG_structure_type)
# CHECK-NEXT:         Atom[2]: 0x00
# CHECK-NEXT:         Atom[3]: {{.*}}
# CHECK-NEXT:       ]
# CHECK-NEXT:       Data 1 [
# CHECK-NEXT:         Atom[0]: 0x00000071
# CHECK-NEXT:         Atom[1]: 0x0013 (DW_TAG_structure_type)
# CHECK:          String: {{.*}} "Outer"
# CHECK-NEXT:       Data 0 [
# CHECK-NEXT:         Atom[0]: {{.*}}
# CHECK-NEXT:         Atom[1]: 0x0013 (DW_TAG_structure_type)

## Generated from:
## namespace N {
## struct Outer {
##   struct Inner {
##     int x;
##   } i;
## };
## } // namespace N
##
## int ez(int x) {
##   struct Inner {
##     int a;
##   } l = {x};
##   return -l.a;
## }
##
## namespace N {
## int FY(Outer o) { return o.i.x + 1; }
## int Ez(Outer o) { return o.i.x; }
## } // namespace N
##
## int Ez(int x) { return x; }
##
## int main() { return N::Ez({}) + N::FY({}) + Ez(1) + ez(1); }
## with: clang++ -target x86_64-apple-macos12 -O0 -g -gdwarf-4 -S

	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z2ezi                         ## -- Begin function _Z2ezi
	.p2align	4
__Z2ezi:                                ## @_Z2ezi
Lfunc_begin0:
	.file	1 "/tmp" "accel.cpp"
	.loc	1 9 0                           ## accel.cpp:9:0
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
Ltmp0:
	.loc	1 12 10 prologue_end            ## accel.cpp:12:10
	movl	-4(%rbp), %eax
	.loc	1 12 9 is_stmt 0                ## accel.cpp:12:9
	movl	%eax, -8(%rbp)
	.loc	1 13 10 is_stmt 1               ## accel.cpp:13:10
	xorl	%eax, %eax
	subl	-8(%rbp), %eax
	.loc	1 13 3 epilogue_begin is_stmt 0 ## accel.cpp:13:3
	popq	%rbp
	retq
Ltmp1:
Lfunc_end0:
	.cfi_endproc
                                        ## -- End function
	.globl	__ZN1N2FYENS_5OuterE            ## -- Begin function _ZN1N2FYENS_5OuterE
	.p2align	4
__ZN1N2FYENS_5OuterE:                   ## @_ZN1N2FYENS_5OuterE
Lfunc_begin1:
	.loc	1 17 0 is_stmt 1                ## accel.cpp:17:0
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
Ltmp2:
	.loc	1 17 30 prologue_end            ## accel.cpp:17:30
	movl	-4(%rbp), %eax
	.loc	1 17 32 is_stmt 0               ## accel.cpp:17:32
	addl	$1, %eax
	.loc	1 17 19 epilogue_begin          ## accel.cpp:17:19
	popq	%rbp
	retq
Ltmp3:
Lfunc_end1:
	.cfi_endproc
                                        ## -- End function
	.globl	__ZN1N2EzENS_5OuterE            ## -- Begin function _ZN1N2EzENS_5OuterE
	.p2align	4
__ZN1N2EzENS_5OuterE:                   ## @_ZN1N2EzENS_5OuterE
Lfunc_begin2:
	.loc	1 18 0 is_stmt 1                ## accel.cpp:18:0
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
Ltmp4:
	.loc	1 18 30 prologue_end            ## accel.cpp:18:30
	movl	-4(%rbp), %eax
	.loc	1 18 19 epilogue_begin is_stmt 0 ## accel.cpp:18:19
	popq	%rbp
	retq
Ltmp5:
Lfunc_end2:
	.cfi_endproc
                                        ## -- End function
	.globl	__Z2Ezi                         ## -- Begin function _Z2Ezi
	.p2align	4
__Z2Ezi:                                ## @_Z2Ezi
Lfunc_begin3:
	.loc	1 21 0 is_stmt 1                ## accel.cpp:21:0
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
Ltmp6:
	.loc	1 21 24 prologue_end            ## accel.cpp:21:24
	movl	-4(%rbp), %eax
	.loc	1 21 17 epilogue_begin is_stmt 0 ## accel.cpp:21:17
	popq	%rbp
	retq
Ltmp7:
Lfunc_end3:
	.cfi_endproc
                                        ## -- End function
	.globl	_main                           ## -- Begin function main
	.p2align	4
_main:                                  ## @main
Lfunc_begin4:
	.loc	1 23 0 is_stmt 1                ## accel.cpp:23:0
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movl	$0, -4(%rbp)
Ltmp8:
	.loc	1 23 28 prologue_end            ## accel.cpp:23:28
	movl	$0, -8(%rbp)
	.loc	1 23 21 is_stmt 0               ## accel.cpp:23:21
	movl	-8(%rbp), %edi
	callq	__ZN1N2EzENS_5OuterE
	movl	%eax, -24(%rbp)                 ## 4-byte Spill
	.loc	1 23 40                         ## accel.cpp:23:40
	movl	$0, -12(%rbp)
	.loc	1 23 33                         ## accel.cpp:23:33
	movl	-12(%rbp), %edi
	callq	__ZN1N2FYENS_5OuterE
	movl	%eax, %ecx
	movl	-24(%rbp), %eax                 ## 4-byte Reload
	.loc	1 23 31                         ## accel.cpp:23:31
	addl	%ecx, %eax
	movl	%eax, -20(%rbp)                 ## 4-byte Spill
	.loc	1 23 45                         ## accel.cpp:23:45
	movl	$1, %edi
	callq	__Z2Ezi
	movl	%eax, %ecx
	movl	-20(%rbp), %eax                 ## 4-byte Reload
	.loc	1 23 43                         ## accel.cpp:23:43
	addl	%ecx, %eax
	movl	%eax, -16(%rbp)                 ## 4-byte Spill
	.loc	1 23 53                         ## accel.cpp:23:53
	movl	$1, %edi
	callq	__Z2ezi
	movl	%eax, %ecx
	movl	-16(%rbp), %eax                 ## 4-byte Reload
	.loc	1 23 51                         ## accel.cpp:23:51
	addl	%ecx, %eax
	.loc	1 23 14 epilogue_begin          ## accel.cpp:23:14
	addq	$32, %rsp
	popq	%rbp
	retq
Ltmp9:
Lfunc_end4:
	.cfi_endproc
                                        ## -- End function
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                               ## Abbreviation Code
	.byte	17                              ## DW_TAG_compile_unit
	.byte	1                               ## DW_CHILDREN_yes
	.byte	37                              ## DW_AT_producer
	.byte	14                              ## DW_FORM_strp
	.byte	19                              ## DW_AT_language
	.byte	5                               ## DW_FORM_data2
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.ascii	"\202|"                         ## DW_AT_LLVM_sysroot
	.byte	14                              ## DW_FORM_strp
	.byte	16                              ## DW_AT_stmt_list
	.byte	23                              ## DW_FORM_sec_offset
	.byte	27                              ## DW_AT_comp_dir
	.byte	14                              ## DW_FORM_strp
	.byte	17                              ## DW_AT_low_pc
	.byte	1                               ## DW_FORM_addr
	.byte	18                              ## DW_AT_high_pc
	.byte	6                               ## DW_FORM_data4
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	2                               ## Abbreviation Code
	.byte	46                              ## DW_TAG_subprogram
	.byte	1                               ## DW_CHILDREN_yes
	.byte	17                              ## DW_AT_low_pc
	.byte	1                               ## DW_FORM_addr
	.byte	18                              ## DW_AT_high_pc
	.byte	6                               ## DW_FORM_data4
	.byte	64                              ## DW_AT_frame_base
	.byte	24                              ## DW_FORM_exprloc
	.byte	110                             ## DW_AT_linkage_name
	.byte	14                              ## DW_FORM_strp
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	63                              ## DW_AT_external
	.byte	25                              ## DW_FORM_flag_present
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	3                               ## Abbreviation Code
	.byte	5                               ## DW_TAG_formal_parameter
	.byte	0                               ## DW_CHILDREN_no
	.byte	2                               ## DW_AT_location
	.byte	24                              ## DW_FORM_exprloc
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	4                               ## Abbreviation Code
	.byte	52                              ## DW_TAG_variable
	.byte	0                               ## DW_CHILDREN_no
	.byte	2                               ## DW_AT_location
	.byte	24                              ## DW_FORM_exprloc
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	5                               ## Abbreviation Code
	.byte	19                              ## DW_TAG_structure_type
	.byte	1                               ## DW_CHILDREN_yes
	.byte	54                              ## DW_AT_calling_convention
	.byte	11                              ## DW_FORM_data1
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	11                              ## DW_AT_byte_size
	.byte	11                              ## DW_FORM_data1
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	6                               ## Abbreviation Code
	.byte	13                              ## DW_TAG_member
	.byte	0                               ## DW_CHILDREN_no
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	56                              ## DW_AT_data_member_location
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	7                               ## Abbreviation Code
	.byte	57                              ## DW_TAG_namespace
	.byte	1                               ## DW_CHILDREN_yes
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	8                               ## Abbreviation Code
	.byte	46                              ## DW_TAG_subprogram
	.byte	0                               ## DW_CHILDREN_no
	.byte	17                              ## DW_AT_low_pc
	.byte	1                               ## DW_FORM_addr
	.byte	18                              ## DW_AT_high_pc
	.byte	6                               ## DW_FORM_data4
	.byte	64                              ## DW_AT_frame_base
	.byte	24                              ## DW_FORM_exprloc
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	63                              ## DW_AT_external
	.byte	25                              ## DW_FORM_flag_present
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	9                               ## Abbreviation Code
	.byte	36                              ## DW_TAG_base_type
	.byte	0                               ## DW_CHILDREN_no
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	62                              ## DW_AT_encoding
	.byte	11                              ## DW_FORM_data1
	.byte	11                              ## DW_AT_byte_size
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	0                               ## EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
Lset0 = Ldebug_info_end0-Ldebug_info_start0 ## Length of Unit
	.long	Lset0
Ldebug_info_start0:
	.short	4                               ## DWARF version number
Lset1 = Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset1
	.byte	8                               ## Address Size (in bytes)
	.byte	1                               ## Abbrev [1] 0xb:0x14a DW_TAG_compile_unit
	.long	0                               ## DW_AT_producer
	.short	33                              ## DW_AT_language
	.long	24                              ## DW_AT_name
	.long	34                              ## DW_AT_LLVM_sysroot
Lset2 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
	.long	Lset2
	.long	36                              ## DW_AT_comp_dir
	.quad	Lfunc_begin0                    ## DW_AT_low_pc
Lset3 = Lfunc_end4-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset3
	.byte	2                               ## Abbrev [2] 0x2e:0x50 DW_TAG_subprogram
	.quad	Lfunc_begin0                    ## DW_AT_low_pc
Lset4 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset4
	.byte	1                               ## DW_AT_frame_base
	.byte	86
	.long	44                              ## DW_AT_linkage_name
	.long	41                              ## DW_AT_name
	.byte	1                               ## DW_AT_decl_file
	.byte	9                               ## DW_AT_decl_line
	.long	333                             ## DW_AT_type
                                        ## DW_AT_external
	.byte	3                               ## Abbrev [3] 0x4b:0xe DW_TAG_formal_parameter
	.byte	2                               ## DW_AT_location
	.byte	145
	.byte	124
	.long	123                             ## DW_AT_name
	.byte	1                               ## DW_AT_decl_file
	.byte	9                               ## DW_AT_decl_line
	.long	333                             ## DW_AT_type
	.byte	4                               ## Abbrev [4] 0x59:0xe DW_TAG_variable
	.byte	2                               ## DW_AT_location
	.byte	145
	.byte	120
	.long	125                             ## DW_AT_name
	.byte	1                               ## DW_AT_decl_file
	.byte	12                              ## DW_AT_decl_line
	.long	103                             ## DW_AT_type
	.byte	5                               ## Abbrev [5] 0x67:0x16 DW_TAG_structure_type
	.byte	5                               ## DW_AT_calling_convention
	.long	111                             ## DW_AT_name
	.byte	4                               ## DW_AT_byte_size
	.byte	1                               ## DW_AT_decl_file
	.byte	10                              ## DW_AT_decl_line
	.byte	6                               ## Abbrev [6] 0x70:0xc DW_TAG_member
	.long	117                             ## DW_AT_name
	.long	333                             ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	11                              ## DW_AT_decl_line
	.byte	0                               ## DW_AT_data_member_location
	.byte	0                               ## End Of Children Mark
	.byte	0                               ## End Of Children Mark
	.byte	7                               ## Abbrev [7] 0x7e:0x8a DW_TAG_namespace
	.long	51                              ## DW_AT_name
	.byte	2                               ## Abbrev [2] 0x83:0x2c DW_TAG_subprogram
	.quad	Lfunc_begin1                    ## DW_AT_low_pc
Lset5 = Lfunc_end1-Lfunc_begin1         ## DW_AT_high_pc
	.long	Lset5
	.byte	1                               ## DW_AT_frame_base
	.byte	86
	.long	56                              ## DW_AT_linkage_name
	.long	53                              ## DW_AT_name
	.byte	1                               ## DW_AT_decl_file
	.byte	17                              ## DW_AT_decl_line
	.long	333                             ## DW_AT_type
                                        ## DW_AT_external
	.byte	3                               ## Abbrev [3] 0xa0:0xe DW_TAG_formal_parameter
	.byte	2                               ## DW_AT_location
	.byte	145
	.byte	124
	.long	127                             ## DW_AT_name
	.byte	1                               ## DW_AT_decl_file
	.byte	17                              ## DW_AT_decl_line
	.long	219                             ## DW_AT_type
	.byte	0                               ## End Of Children Mark
	.byte	2                               ## Abbrev [2] 0xaf:0x2c DW_TAG_subprogram
	.quad	Lfunc_begin2                    ## DW_AT_low_pc
Lset6 = Lfunc_end2-Lfunc_begin2         ## DW_AT_high_pc
	.long	Lset6
	.byte	1                               ## DW_AT_frame_base
	.byte	86
	.long	79                              ## DW_AT_linkage_name
	.long	76                              ## DW_AT_name
	.byte	1                               ## DW_AT_decl_file
	.byte	18                              ## DW_AT_decl_line
	.long	333                             ## DW_AT_type
                                        ## DW_AT_external
	.byte	3                               ## Abbrev [3] 0xcc:0xe DW_TAG_formal_parameter
	.byte	2                               ## DW_AT_location
	.byte	145
	.byte	124
	.long	127                             ## DW_AT_name
	.byte	1                               ## DW_AT_decl_file
	.byte	18                              ## DW_AT_decl_line
	.long	219                             ## DW_AT_type
	.byte	0                               ## End Of Children Mark
	.byte	5                               ## Abbrev [5] 0xdb:0x2c DW_TAG_structure_type
	.byte	5                               ## DW_AT_calling_convention
	.long	129                             ## DW_AT_name
	.byte	4                               ## DW_AT_byte_size
	.byte	1                               ## DW_AT_decl_file
	.byte	2                               ## DW_AT_decl_line
	.byte	6                               ## Abbrev [6] 0xe4:0xc DW_TAG_member
	.long	135                             ## DW_AT_name
	.long	240                             ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	5                               ## DW_AT_decl_line
	.byte	0                               ## DW_AT_data_member_location
	.byte	5                               ## Abbrev [5] 0xf0:0x16 DW_TAG_structure_type
	.byte	5                               ## DW_AT_calling_convention
	.long	111                             ## DW_AT_name
	.byte	4                               ## DW_AT_byte_size
	.byte	1                               ## DW_AT_decl_file
	.byte	3                               ## DW_AT_decl_line
	.byte	6                               ## Abbrev [6] 0xf9:0xc DW_TAG_member
	.long	123                             ## DW_AT_name
	.long	333                             ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	4                               ## DW_AT_decl_line
	.byte	0                               ## DW_AT_data_member_location
	.byte	0                               ## End Of Children Mark
	.byte	0                               ## End Of Children Mark
	.byte	0                               ## End Of Children Mark
	.byte	2                               ## Abbrev [2] 0x108:0x2c DW_TAG_subprogram
	.quad	Lfunc_begin3                    ## DW_AT_low_pc
Lset7 = Lfunc_end3-Lfunc_begin3         ## DW_AT_high_pc
	.long	Lset7
	.byte	1                               ## DW_AT_frame_base
	.byte	86
	.long	99                              ## DW_AT_linkage_name
	.long	76                              ## DW_AT_name
	.byte	1                               ## DW_AT_decl_file
	.byte	21                              ## DW_AT_decl_line
	.long	333                             ## DW_AT_type
                                        ## DW_AT_external
	.byte	3                               ## Abbrev [3] 0x125:0xe DW_TAG_formal_parameter
	.byte	2                               ## DW_AT_location
	.byte	145
	.byte	124
	.long	123                             ## DW_AT_name
	.byte	1                               ## DW_AT_decl_file
	.byte	21                              ## DW_AT_decl_line
	.long	333                             ## DW_AT_type
	.byte	0                               ## End Of Children Mark
	.byte	8                               ## Abbrev [8] 0x134:0x19 DW_TAG_subprogram
	.quad	Lfunc_begin4                    ## DW_AT_low_pc
Lset8 = Lfunc_end4-Lfunc_begin4         ## DW_AT_high_pc
	.long	Lset8
	.byte	1                               ## DW_AT_frame_base
	.byte	86
	.long	106                             ## DW_AT_name
	.byte	1                               ## DW_AT_decl_file
	.byte	23                              ## DW_AT_decl_line
	.long	333                             ## DW_AT_type
                                        ## DW_AT_external
	.byte	9                               ## Abbrev [9] 0x14d:0x7 DW_TAG_base_type
	.long	119                             ## DW_AT_name
	.byte	5                               ## DW_AT_encoding
	.byte	4                               ## DW_AT_byte_size
	.byte	0                               ## End Of Children Mark
Ldebug_info_end0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang version 24.0.0git"       ## string offset=0 ; clang version 24.0.0git
	.asciz	"accel.cpp"                     ## string offset=24 ; accel.cpp
	.asciz	"/"                             ## string offset=34 ; /
	.asciz	"/tmp"                          ## string offset=36 ; /tmp
	.asciz	"ez"                            ## string offset=41 ; ez
	.asciz	"_Z2ezi"                        ## string offset=44 ; _Z2ezi
	.asciz	"N"                             ## string offset=51 ; N
	.asciz	"FY"                            ## string offset=53 ; FY
	.asciz	"_ZN1N2FYENS_5OuterE"           ## string offset=56 ; _ZN1N2FYENS_5OuterE
	.asciz	"Ez"                            ## string offset=76 ; Ez
	.asciz	"_ZN1N2EzENS_5OuterE"           ## string offset=79 ; _ZN1N2EzENS_5OuterE
	.asciz	"_Z2Ezi"                        ## string offset=99 ; _Z2Ezi
	.asciz	"main"                          ## string offset=106 ; main
	.asciz	"Inner"                         ## string offset=111 ; Inner
	.asciz	"a"                             ## string offset=117 ; a
	.asciz	"int"                           ## string offset=119 ; int
	.asciz	"x"                             ## string offset=123 ; x
	.asciz	"l"                             ## string offset=125 ; l
	.asciz	"o"                             ## string offset=127 ; o
	.asciz	"Outer"                         ## string offset=129 ; Outer
	.asciz	"i"                             ## string offset=135 ; i
	.section	__DWARF,__apple_names,regular,debug
Lnames_begin:
	.long	1212240712                      ## Header Magic
	.short	1                               ## Header Version
	.short	0                               ## Header Hash Function
	.long	6                               ## Header Bucket Count
	.long	6                               ## Header Hash Count
	.long	12                              ## Header Data Length
	.long	0                               ## HeaderData Die Offset Base
	.long	1                               ## HeaderData Atom Count
	.short	1                               ## DW_ATOM_die_offset
	.short	6                               ## DW_FORM_data4
	.long	-1                              ## Bucket 0
	.long	-1                              ## Bucket 1
	.long	0                               ## Bucket 2
	.long	-1                              ## Bucket 3
	.long	5                               ## Bucket 4
	.long	-1                              ## Bucket 5
	.long	5862308                         ## Hash in Bucket 2
	.long	5863364                         ## Hash in Bucket 2
	.long	-336439688                      ## Hash in Bucket 2
	.long	-336404840                      ## Hash in Bucket 2
	.long	-160166966                      ## Hash in Bucket 2
	.long	2090499946                      ## Hash in Bucket 4
Lset9 = LNames0-Lnames_begin            ## Offset in Bucket 2
	.long	Lset9
Lset10 = LNames2-Lnames_begin           ## Offset in Bucket 2
	.long	Lset10
Lset11 = LNames3-Lnames_begin           ## Offset in Bucket 2
	.long	Lset11
Lset12 = LNames4-Lnames_begin           ## Offset in Bucket 2
	.long	Lset12
Lset13 = LNames5-Lnames_begin           ## Offset in Bucket 2
	.long	Lset13
Lset14 = LNames7-Lnames_begin           ## Offset in Bucket 4
	.long	Lset14
LNames0:
	.long	76                              ## Ez
	.long	2                               ## Num DIEs
	.long	175
	.long	264
LNames1:
	.long	53                              ## FY
	.long	1                               ## Num DIEs
	.long	131
	.long	0
LNames2:
	.long	41                              ## ez
	.long	1                               ## Num DIEs
	.long	46
	.long	0
LNames3:
	.long	99                              ## _Z2Ezi
	.long	1                               ## Num DIEs
	.long	264
	.long	0
LNames4:
	.long	44                              ## _Z2ezi
	.long	1                               ## Num DIEs
	.long	46
	.long	0
LNames5:
	.long	79                              ## _ZN1N2EzENS_5OuterE
	.long	1                               ## Num DIEs
	.long	175
LNames6:
	.long	56                              ## _ZN1N2FYENS_5OuterE
	.long	1                               ## Num DIEs
	.long	131
	.long	0
LNames7:
	.long	106                             ## main
	.long	1                               ## Num DIEs
	.long	308
	.long	0
	.section	__DWARF,__apple_objc,regular,debug
Lobjc_begin:
	.long	1212240712                      ## Header Magic
	.short	1                               ## Header Version
	.short	0                               ## Header Hash Function
	.long	1                               ## Header Bucket Count
	.long	0                               ## Header Hash Count
	.long	12                              ## Header Data Length
	.long	0                               ## HeaderData Die Offset Base
	.long	1                               ## HeaderData Atom Count
	.short	1                               ## DW_ATOM_die_offset
	.short	6                               ## DW_FORM_data4
	.long	-1                              ## Bucket 0
	.section	__DWARF,__apple_namespac,regular,debug
Lnamespac_begin:
	.long	1212240712                      ## Header Magic
	.short	1                               ## Header Version
	.short	0                               ## Header Hash Function
	.long	1                               ## Header Bucket Count
	.long	1                               ## Header Hash Count
	.long	12                              ## Header Data Length
	.long	0                               ## HeaderData Die Offset Base
	.long	1                               ## HeaderData Atom Count
	.short	1                               ## DW_ATOM_die_offset
	.short	6                               ## DW_FORM_data4
	.long	0                               ## Bucket 0
	.long	177651                          ## Hash in Bucket 0
Lset15 = Lnamespac0-Lnamespac_begin     ## Offset in Bucket 0
	.long	Lset15
Lnamespac0:
	.long	51                              ## N
	.long	1                               ## Num DIEs
	.long	126
	.long	0
	.section	__DWARF,__apple_types,regular,debug
Ltypes_begin:
	.long	1212240712                      ## Header Magic
	.short	1                               ## Header Version
	.short	0                               ## Header Hash Function
	.long	3                               ## Header Bucket Count
	.long	3                               ## Header Hash Count
	.long	20                              ## Header Data Length
	.long	0                               ## HeaderData Die Offset Base
	.long	3                               ## HeaderData Atom Count
	.short	1                               ## DW_ATOM_die_offset
	.short	6                               ## DW_FORM_data4
	.short	3                               ## DW_ATOM_die_tag
	.short	5                               ## DW_FORM_data2
	.short	4                               ## DW_ATOM_type_flags
	.short	11                              ## DW_FORM_data1
	.long	-1                              ## Bucket 0
	.long	-1                              ## Bucket 1
	.long	0                               ## Bucket 2
	.long	193495088                       ## Hash in Bucket 2
	.long	224800769                       ## Hash in Bucket 2
	.long	232174388                       ## Hash in Bucket 2
Lset16 = Ltypes0-Ltypes_begin           ## Offset in Bucket 2
	.long	Lset16
Lset17 = Ltypes1-Ltypes_begin           ## Offset in Bucket 2
	.long	Lset17
Lset18 = Ltypes2-Ltypes_begin           ## Offset in Bucket 2
	.long	Lset18
Ltypes0:
	.long	119                             ## int
	.long	1                               ## Num DIEs
	.long	333
	.short	36
	.byte	0
	.long	0
Ltypes1:
	.long	111                             ## Inner
	.long	2                               ## Num DIEs
	.long	103
	.short	19
	.byte	0
	.long	240
	.short	19
	.byte	0
	.long	0
Ltypes2:
	.long	129                             ## Outer
	.long	1                               ## Num DIEs
	.long	219
	.short	19
	.byte	0
	.long	0
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
