# REQUIRES: bpf-registered-target

## Verify that when llvm-objdump uses .BTF.ext to show CO-RE
## relocations formatting options operate as expected.

# RUN: llvm-mc --triple bpfel %s --filetype=obj | \
# RUN:   llvm-objdump --no-addresses --no-show-raw-insn -dr - | \
# RUN:   FileCheck --strict-whitespace --check-prefix=NOADDR %s

# RUN: llvm-mc --triple bpfel %s --filetype=obj | \
# RUN:   llvm-objdump --no-addresses --no-show-raw-insn -d - | \
# RUN:   FileCheck --strict-whitespace --check-prefix=NORELO %s

# RUN: llvm-mc --triple bpfel %s --filetype=obj | \
# RUN:   llvm-objdump --no-show-raw-insn -dr - | \
# RUN:   FileCheck --strict-whitespace --check-prefix=ADDR %s

# RUN: llvm-mc --triple bpfel %s --filetype=obj | \
# RUN:   llvm-objdump --adjust-vma=0x10 --no-show-raw-insn -dr - | \
# RUN:   FileCheck --strict-whitespace --check-prefix=VMA %s

## Input generated from the following C code:
##
##   #define __pai __attribute__((preserve_access_index))
##   struct foo {
##     int a;
##   } __pai;
##   enum bar { U, V };
##   extern void consume(unsigned long);
##   void root() {
##     asm volatile("r0 = 42;":::);
##     struct foo *foo = 0;
##     consume(__builtin_preserve_type_info(*foo, 0));
##     consume((unsigned long) &foo->a);
##     consume(__builtin_preserve_enum_value(*(enum bar *)U, 0));
##   }
##
## Using the following command:
##
##  clang -target bpf -g -O2 -S t.c

# NOADDR:	r1 = 0x1
# NOADDR-NEXT:		CO-RE <type_exists> [3] struct foo
# NOADDR-NEXT:	call -0x1
# NOADDR-NEXT:		R_BPF_64_32	consume
# NOADDR-NEXT:	r1 = 0x0
# NOADDR-NEXT:		CO-RE <byte_off> [3] struct foo::a (0:0)
# NOADDR-NEXT:	call -0x1
# NOADDR-NEXT:		R_BPF_64_32	consume
# NOADDR-NEXT:	r1 = 0x1 ll
# NOADDR-NEXT:		CO-RE <enumval_exists> [8] enum bar::U = 0
# NOADDR-NEXT:	call -0x1
# NOADDR-NEXT:		R_BPF_64_32	consume
# NOADDR-NEXT:	exit

# NORELO:	r1 = 0x1
# NORELO-NEXT:	call -0x1
# NORELO-NEXT:	r1 = 0x0
# NORELO-NEXT:	call -0x1
# NORELO-NEXT:	r1 = 0x1 ll
# NORELO-NEXT:	call -0x1
# NORELO-NEXT:	exit

# ADDR:            1:	r1 = 0x1
# ADDR-NEXT:		0000000000000008:  CO-RE <type_exists> [3] struct foo
# ADDR-NEXT:       2:	call -0x1
# ADDR-NEXT:		0000000000000010:  R_BPF_64_32	consume
# ADDR-NEXT:       3:	r1 = 0x0
# ADDR-NEXT:		0000000000000018:  CO-RE <byte_off> [3] struct foo::a (0:0)
# ADDR-NEXT:       4:	call -0x1
# ADDR-NEXT:		0000000000000020:  R_BPF_64_32	consume
# ADDR-NEXT:       5:	r1 = 0x1 ll
# ADDR-NEXT:		0000000000000028:  CO-RE <enumval_exists> [8] enum bar::U = 0
# ADDR-NEXT:       7:	call -0x1
# ADDR-NEXT:		0000000000000038:  R_BPF_64_32	consume
# ADDR-NEXT:       8:	exit

# VMA:            3:	r1 = 0x1
# VMA-NEXT:		0000000000000018:  CO-RE <type_exists> [3] struct foo
# VMA-NEXT:       4:	call -0x1
# VMA-NEXT:		0000000000000010:  R_BPF_64_32	consume
# VMA-NEXT:       5:	r1 = 0x0
# VMA-NEXT:		0000000000000028:  CO-RE <byte_off> [3] struct foo::a (0:0)
# VMA-NEXT:       6:	call -0x1
# VMA-NEXT:		0000000000000020:  R_BPF_64_32	consume
# VMA-NEXT:       7:	r1 = 0x1 ll
# VMA-NEXT:		0000000000000038:  CO-RE <enumval_exists> [8] enum bar::U = 0
# VMA-NEXT:       9:	call -0x1
# VMA-NEXT:		0000000000000038:  R_BPF_64_32	consume
# VMA-NEXT:      10:	exit

	.text
	.file	"t.c"
	.file	0 "/home/eddy/work/tmp" "t.c" md5 0x7675be79a30f35c69b89cf826ff55a5f
	.globl	root                    # -- Begin function root
	.p2align	3
	.type	root,@function
root:                                   # @root
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:                                # %entry
	.loc	0 8 3 prologue_end      # t.c:8:3
.Ltmp0:
	#APP
	r0 = 42

	#NO_APP
.Ltmp1:
.Ltmp2:
	#DEBUG_VALUE: root:foo <- 0
	.loc	0 10 3                  # t.c:10:3
.Ltmp3:
.Ltmp4:
	r1 = 1
	call consume
.Ltmp5:
	.loc	0 11 3                  # t.c:11:3
.Ltmp6:
.Ltmp7:
	r1 = 0
	call consume
.Ltmp8:
	.loc	0 12 3                  # t.c:12:3
.Ltmp9:
.Ltmp10:
	r1 = 1 ll
	call consume
.Ltmp11:
	.loc	0 13 1                  # t.c:13:1
.Ltmp12:
	exit
.Ltmp13:
.Ltmp14:
.Lfunc_end0:
	.size	root, .Lfunc_end0-root
	.cfi_endproc
                                        # -- End function
	.section	.debug_loclists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
	.short	5                       # Version
	.byte	8                       # Address size
	.byte	0                       # Segment selector size
	.long	1                       # Offset entry count
.Lloclists_table_base0:
	.long	.Ldebug_loc0-.Lloclists_table_base0
.Ldebug_loc0:
	.byte	4                       # DW_LLE_offset_pair
	.uleb128 .Ltmp1-.Lfunc_begin0   #   starting offset
	.uleb128 .Lfunc_end0-.Lfunc_begin0 #   ending offset
	.byte	2                       # Loc expr size
	.byte	48                      # DW_OP_lit0
	.byte	159                     # DW_OP_stack_value
	.byte	0                       # DW_LLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	37                      # DW_FORM_strx1
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	114                     # DW_AT_str_offsets_base
	.byte	23                      # DW_FORM_sec_offset
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
	.byte	27                      # DW_AT_comp_dir
	.byte	37                      # DW_FORM_strx1
	.byte	17                      # DW_AT_low_pc
	.byte	27                      # DW_FORM_addrx
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	115                     # DW_AT_addr_base
	.byte	23                      # DW_FORM_sec_offset
	.ascii	"\214\001"              # DW_AT_loclists_base
	.byte	23                      # DW_FORM_sec_offset
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	4                       # DW_TAG_enumeration_type
	.byte	1                       # DW_CHILDREN_yes
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	40                      # DW_TAG_enumerator
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	28                      # DW_AT_const_value
	.byte	15                      # DW_FORM_udata
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	4                       # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	5                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	27                      # DW_FORM_addrx
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	64                      # DW_AT_frame_base
	.byte	24                      # DW_FORM_exprloc
	.byte	122                     # DW_AT_call_all_calls
	.byte	25                      # DW_FORM_flag_present
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	6                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	34                      # DW_FORM_loclistx
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	7                       # Abbreviation Code
	.byte	72                      # DW_TAG_call_site
	.byte	0                       # DW_CHILDREN_no
	.byte	127                     # DW_AT_call_origin
	.byte	19                      # DW_FORM_ref4
	.byte	125                     # DW_AT_call_return_pc
	.byte	27                      # DW_FORM_addrx
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	8                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	39                      # DW_AT_prototyped
	.byte	25                      # DW_FORM_flag_present
	.byte	60                      # DW_AT_declaration
	.byte	25                      # DW_FORM_flag_present
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	9                       # Abbreviation Code
	.byte	5                       # DW_TAG_formal_parameter
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	10                      # Abbreviation Code
	.byte	15                      # DW_TAG_pointer_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	11                      # Abbreviation Code
	.byte	19                      # DW_TAG_structure_type
	.byte	1                       # DW_CHILDREN_yes
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	12                      # Abbreviation Code
	.byte	13                      # DW_TAG_member
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	56                      # DW_AT_data_member_location
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                       # DWARF version number
	.byte	1                       # DWARF Unit Type
	.byte	8                       # Address Size (in bytes)
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	1                       # Abbrev [1] 0xc:0x7d DW_TAG_compile_unit
	.byte	0                       # DW_AT_producer
	.short	29                      # DW_AT_language
	.byte	1                       # DW_AT_name
	.long	.Lstr_offsets_base0     # DW_AT_str_offsets_base
	.long	.Lline_table_start0     # DW_AT_stmt_list
	.byte	2                       # DW_AT_comp_dir
	.byte	0                       # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.long	.Laddr_table_base0      # DW_AT_addr_base
	.long	.Lloclists_table_base0  # DW_AT_loclists_base
	.byte	2                       # Abbrev [2] 0x27:0x10 DW_TAG_enumeration_type
	.long	55                      # DW_AT_type
	.byte	6                       # DW_AT_name
	.byte	4                       # DW_AT_byte_size
	.byte	0                       # DW_AT_decl_file
	.byte	5                       # DW_AT_decl_line
	.byte	3                       # Abbrev [3] 0x30:0x3 DW_TAG_enumerator
	.byte	4                       # DW_AT_name
	.byte	0                       # DW_AT_const_value
	.byte	3                       # Abbrev [3] 0x33:0x3 DW_TAG_enumerator
	.byte	5                       # DW_AT_name
	.byte	1                       # DW_AT_const_value
	.byte	0                       # End Of Children Mark
	.byte	4                       # Abbrev [4] 0x37:0x4 DW_TAG_base_type
	.byte	3                       # DW_AT_name
	.byte	7                       # DW_AT_encoding
	.byte	4                       # DW_AT_byte_size
	.byte	4                       # Abbrev [4] 0x3b:0x4 DW_TAG_base_type
	.byte	7                       # DW_AT_name
	.byte	7                       # DW_AT_encoding
	.byte	8                       # DW_AT_byte_size
	.byte	5                       # Abbrev [5] 0x3f:0x27 DW_TAG_subprogram
	.byte	0                       # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	90
                                        # DW_AT_call_all_calls
	.byte	9                       # DW_AT_name
	.byte	0                       # DW_AT_decl_file
	.byte	7                       # DW_AT_decl_line
                                        # DW_AT_external
	.byte	6                       # Abbrev [6] 0x4a:0x9 DW_TAG_variable
	.byte	0                       # DW_AT_location
	.byte	10                      # DW_AT_name
	.byte	0                       # DW_AT_decl_file
	.byte	9                       # DW_AT_decl_line
	.long	112                     # DW_AT_type
	.byte	7                       # Abbrev [7] 0x53:0x6 DW_TAG_call_site
	.long	102                     # DW_AT_call_origin
	.byte	1                       # DW_AT_call_return_pc
	.byte	7                       # Abbrev [7] 0x59:0x6 DW_TAG_call_site
	.long	102                     # DW_AT_call_origin
	.byte	2                       # DW_AT_call_return_pc
	.byte	7                       # Abbrev [7] 0x5f:0x6 DW_TAG_call_site
	.long	102                     # DW_AT_call_origin
	.byte	3                       # DW_AT_call_return_pc
	.byte	0                       # End Of Children Mark
	.byte	8                       # Abbrev [8] 0x66:0xa DW_TAG_subprogram
	.byte	8                       # DW_AT_name
	.byte	0                       # DW_AT_decl_file
	.byte	6                       # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	9                       # Abbrev [9] 0x6a:0x5 DW_TAG_formal_parameter
	.long	59                      # DW_AT_type
	.byte	0                       # End Of Children Mark
	.byte	10                      # Abbrev [10] 0x70:0x5 DW_TAG_pointer_type
	.long	117                     # DW_AT_type
	.byte	11                      # Abbrev [11] 0x75:0xf DW_TAG_structure_type
	.byte	10                      # DW_AT_name
	.byte	4                       # DW_AT_byte_size
	.byte	0                       # DW_AT_decl_file
	.byte	2                       # DW_AT_decl_line
	.byte	12                      # Abbrev [12] 0x7a:0x9 DW_TAG_member
	.byte	11                      # DW_AT_name
	.long	132                     # DW_AT_type
	.byte	0                       # DW_AT_decl_file
	.byte	3                       # DW_AT_decl_line
	.byte	0                       # DW_AT_data_member_location
	.byte	0                       # End Of Children Mark
	.byte	4                       # Abbrev [4] 0x84:0x4 DW_TAG_base_type
	.byte	12                      # DW_AT_name
	.byte	5                       # DW_AT_encoding
	.byte	4                       # DW_AT_byte_size
	.byte	0                       # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	56                      # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 17.0.0 (/home/eddy/work/llvm-project/clang 76d673bb89f8ec8cf65a4294a98a83c9d6646b11)" # string offset=0
.Linfo_string1:
	.asciz	"t.c"                   # string offset=99
.Linfo_string2:
	.asciz	"/home/eddy/work/tmp"   # string offset=103
.Linfo_string3:
	.asciz	"unsigned int"          # string offset=123
.Linfo_string4:
	.asciz	"U"                     # string offset=136
.Linfo_string5:
	.asciz	"V"                     # string offset=138
.Linfo_string6:
	.asciz	"bar"                   # string offset=140
.Linfo_string7:
	.asciz	"unsigned long"         # string offset=144
.Linfo_string8:
	.asciz	"consume"               # string offset=158
.Linfo_string9:
	.asciz	"root"                  # string offset=166
.Linfo_string10:
	.asciz	"foo"                   # string offset=171
.Linfo_string11:
	.asciz	"a"                     # string offset=175
.Linfo_string12:
	.asciz	"int"                   # string offset=177
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string11
	.long	.Linfo_string12
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                       # DWARF version number
	.byte	8                       # Address size
	.byte	0                       # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Ltmp5
	.quad	.Ltmp8
	.quad	.Ltmp11
.Ldebug_addr_end0:
	.section	.BTF,"",@progbits
	.short	60319                   # 0xeb9f
	.byte	1
	.byte	0
	.long	24
	.long	0
	.long	140
	.long	140
	.long	262
	.long	0                       # BTF_KIND_FUNC_PROTO(id = 1)
	.long	218103808               # 0xd000000
	.long	0
	.long	1                       # BTF_KIND_FUNC(id = 2)
	.long	201326593               # 0xc000001
	.long	1
	.long	67                      # BTF_KIND_STRUCT(id = 3)
	.long	67108865                # 0x4000001
	.long	4
	.long	71
	.long	4
	.long	0                       # 0x0
	.long	73                      # BTF_KIND_INT(id = 4)
	.long	16777216                # 0x1000000
	.long	4
	.long	16777248                # 0x1000020
	.long	0                       # BTF_KIND_FUNC_PROTO(id = 5)
	.long	218103809               # 0xd000001
	.long	0
	.long	0
	.long	6
	.long	129                     # BTF_KIND_INT(id = 6)
	.long	16777216                # 0x1000000
	.long	8
	.long	64                      # 0x40
	.long	143                     # BTF_KIND_FUNC(id = 7)
	.long	201326594               # 0xc000002
	.long	5
	.long	191                     # BTF_KIND_ENUM(id = 8)
	.long	100663298               # 0x6000002
	.long	4
	.long	195
	.long	0
	.long	197
	.long	1
	.byte	0                       # string offset=0
	.ascii	"root"                  # string offset=1
	.byte	0
	.ascii	".text"                 # string offset=6
	.byte	0
	.ascii	"/home/eddy/work/tmp/t.c" # string offset=12
	.byte	0
	.ascii	"  asm volatile(\"r0 = 42;\":::);" # string offset=36
	.byte	0
	.ascii	"foo"                   # string offset=67
	.byte	0
	.byte	97                      # string offset=71
	.byte	0
	.ascii	"int"                   # string offset=73
	.byte	0
	.byte	48                      # string offset=77
	.byte	0
	.ascii	"  consume(__builtin_preserve_type_info(*foo, 0));" # string offset=79
	.byte	0
	.ascii	"unsigned long"         # string offset=129
	.byte	0
	.ascii	"consume"               # string offset=143
	.byte	0
	.ascii	"0:0"                   # string offset=151
	.byte	0
	.ascii	"  consume((unsigned long) &foo->a);" # string offset=155
	.byte	0
	.ascii	"bar"                   # string offset=191
	.byte	0
	.byte	85                      # string offset=195
	.byte	0
	.byte	86                      # string offset=197
	.byte	0
	.ascii	"  consume(__builtin_preserve_enum_value(*(enum bar *)U, 0));" # string offset=199
	.byte	0
	.byte	125                     # string offset=260
	.byte	0
	.section	.BTF.ext,"",@progbits
	.short	60319                   # 0xeb9f
	.byte	1
	.byte	0
	.long	32
	.long	0
	.long	20
	.long	20
	.long	92
	.long	112
	.long	60
	.long	8                       # FuncInfo
	.long	6                       # FuncInfo section string offset=6
	.long	1
	.long	.Lfunc_begin0
	.long	2
	.long	16                      # LineInfo
	.long	6                       # LineInfo section string offset=6
	.long	5
	.long	.Ltmp0
	.long	12
	.long	36
	.long	8195                    # Line 8 Col 3
	.long	.Ltmp4
	.long	12
	.long	79
	.long	10243                   # Line 10 Col 3
	.long	.Ltmp7
	.long	12
	.long	155
	.long	11267                   # Line 11 Col 3
	.long	.Ltmp10
	.long	12
	.long	199
	.long	12291                   # Line 12 Col 3
	.long	.Ltmp12
	.long	12
	.long	260
	.long	13313                   # Line 13 Col 1
	.long	16                      # FieldReloc
	.long	6                       # Field reloc section string offset=6
	.long	3
	.long	.Ltmp3
	.long	3
	.long	77
	.long	8
	.long	.Ltmp6
	.long	3
	.long	151
	.long	0
	.long	.Ltmp9
	.long	8
	.long	77
	.long	10
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
