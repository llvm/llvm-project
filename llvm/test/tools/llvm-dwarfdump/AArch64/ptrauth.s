# RUN: llvm-mc %s -filetype obj -triple arm64-apple-darwin -o %t
# RUN: llvm-dwarfdump --debug-info %t | FileCheck %s
# RUN: llvm-dwarfdump --verify %t

# CHECK: 0x0000001e:   DW_TAG_variable
# CHECK:                 DW_AT_name      ("p1")
# CHECK:                 DW_AT_type      (0x00000033 "void *__ptrauth(4, 1, 0x04d2)")

# CHECK: 0x00000033:   DW_TAG_LLVM_ptrauth_type
# CHECK:                 DW_AT_LLVM_ptrauth_key  (0x04)
# CHECK:                 DW_AT_LLVM_ptrauth_address_discriminated        (true)
# CHECK:                 DW_AT_LLVM_ptrauth_extra_discriminator  (0x04d2)

# CHECK: 0x0000003c:   DW_TAG_variable
# CHECK:                 DW_AT_name      ("p2")
# CHECK:                 DW_AT_type      (0x00000047 "void *__ptrauth(4, 1, 0x04d3, "isa-pointer")")

# CHECK: 0x00000047:   DW_TAG_LLVM_ptrauth_type
# CHECK:                 DW_AT_LLVM_ptrauth_key  (0x04)
# CHECK:                 DW_AT_LLVM_ptrauth_address_discriminated        (true)
# CHECK:                 DW_AT_LLVM_ptrauth_extra_discriminator  (0x04d3)
# CHECK:                 DW_AT_LLVM_ptrauth_isa_pointer  (true)

# CHECK: 0x0000004f:   DW_TAG_variable
# CHECK:                 DW_AT_name      ("p3")
# CHECK:                 DW_AT_type      (0x0000005a "void *__ptrauth(4, 1, 0x04d4, "authenticates-null-values,strip")")

# CHECK: 0x0000005a:   DW_TAG_LLVM_ptrauth_type
# CHECK:                 DW_AT_LLVM_ptrauth_key  (0x04)
# CHECK:                 DW_AT_LLVM_ptrauth_address_discriminated        (true)
# CHECK:                 DW_AT_LLVM_ptrauth_extra_discriminator  (0x04d4)
# CHECK:                 DW_AT_LLVM_ptrauth_authenticates_null_values    (true)

# CHECK: 0x00000063:   DW_TAG_variable
# CHECK:                 DW_AT_name      ("p4")
# CHECK:                 DW_AT_type (0x0000006e "void *__ptrauth(4, 1, 0x04d5, "isa-pointer,authenticates-null-values,sign-and-strip")")

# CHECK: 0x0000006e:   DW_TAG_LLVM_ptrauth_type
# CHECK:                 DW_AT_LLVM_ptrauth_key  (0x04)
# CHECK:                 DW_AT_LLVM_ptrauth_address_discriminated        (true)
# CHECK:                 DW_AT_LLVM_ptrauth_extra_discriminator  (0x04d5)
# CHECK:                 DW_AT_LLVM_ptrauth_isa_pointer  (true)
# CHECK:                 DW_AT_LLVM_ptrauth_authenticates_null_values    (true)

	.section	__TEXT,__text,regular,pure_instructions
	.file	1 "/" "/tmp/p.c"
	.comm	_p,8,3                          ; @p
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	52                              ; DW_TAG_variable
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	2                               ; DW_AT_location
	.byte	24                              ; DW_FORM_exprloc
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.ascii	"\200\206\001"                  ; DW_TAG_LLVM_ptrauth_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.ascii	"\204|"                         ; DW_AT_LLVM_ptrauth_key
	.byte	11                              ; DW_FORM_data1
	.ascii	"\205|"                         ; DW_AT_LLVM_ptrauth_address_discriminated
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\206|"                         ; DW_AT_LLVM_ptrauth_extra_discriminator
	.byte	5                               ; DW_FORM_data2
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	15                              ; DW_TAG_pointer_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	52                              ; DW_TAG_variable
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.ascii	"\200\206\001"                  ; DW_TAG_LLVM_ptrauth_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.ascii	"\204|"                         ; DW_AT_LLVM_ptrauth_key
	.byte	11                              ; DW_FORM_data1
	.ascii	"\205|"                         ; DW_AT_LLVM_ptrauth_address_discriminated
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\206|"                         ; DW_AT_LLVM_ptrauth_extra_discriminator
	.byte	5                               ; DW_FORM_data2
	.ascii	"\210|"                         ; DW_AT_LLVM_ptrauth_isa_pointer
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
	.ascii	"\200\206\001"                  ; DW_TAG_LLVM_ptrauth_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.ascii	"\204|"                         ; DW_AT_LLVM_ptrauth_key
	.byte	11                              ; DW_FORM_data1
	.ascii	"\205|"                         ; DW_AT_LLVM_ptrauth_address_discriminated
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\206|"                         ; DW_AT_LLVM_ptrauth_extra_discriminator
	.byte	5                               ; DW_FORM_data2
	.ascii	"\211|"                         ; DW_AT_LLVM_ptrauth_authenticates_null_values
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\212|"                         ; DW_AT_LLVM_ptrauth_authentication_mode
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	8                               ; Abbreviation Code
	.ascii	"\200\206\001"                  ; DW_TAG_LLVM_ptrauth_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.ascii	"\204|"                         ; DW_AT_LLVM_ptrauth_key
	.byte	11                              ; DW_FORM_data1
	.ascii	"\205|"                         ; DW_AT_LLVM_ptrauth_address_discriminated
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\206|"                         ; DW_AT_LLVM_ptrauth_extra_discriminator
	.byte	5                               ; DW_FORM_data2
	.ascii	"\210|"                         ; DW_AT_LLVM_ptrauth_isa_pointer
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\211|"                         ; DW_AT_LLVM_ptrauth_authenticates_null_values
	.byte	25                              ; DW_FORM_flag_present
	.ascii	"\212|"                         ; DW_AT_LLVM_ptrauth_authentication_mode
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
.set Lset0, Ldebug_info_end0-Ldebug_info_start0 ; Length of Unit
	.long	Lset0
Ldebug_info_start0:
	.short	4                               ; DWARF version number
.set Lset1, Lsection_abbrev-Lsection_abbrev ; Offset Into Abbrev. Section
	.long	Lset1
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x6d DW_TAG_compile_unit
	.long	0                               ; DW_AT_producer
	.short	12                              ; DW_AT_language
	.long	1                               ; DW_AT_name
.set Lset2, Lline_table_start0-Lsection_line ; DW_AT_stmt_list
	.long	Lset2
	.long	10                              ; DW_AT_comp_dir
	.byte	2                               ; Abbrev [2] 0x1e:0x15 DW_TAG_variable
	.long	12                              ; DW_AT_name
	.long	51                              ; DW_AT_type
                                        ; DW_AT_external
	.byte	1                               ; DW_AT_decl_file
	.byte	1                               ; DW_AT_decl_line
	.byte	9                               ; DW_AT_location
	.byte	3
	.quad	_p
	.byte	3                               ; Abbrev [3] 0x33:0x8 DW_TAG_LLVM_ptrauth_type
	.long	59                              ; DW_AT_type
	.byte	4                               ; DW_AT_LLVM_ptrauth_key
                                        ; DW_AT_LLVM_ptrauth_address_discriminated
	.short	1234                            ; DW_AT_LLVM_ptrauth_extra_discriminator
	.byte	4                               ; Abbrev [4] 0x3b:0x1 DW_TAG_pointer_type
	.byte	5                               ; Abbrev [5] 0x3c:0xb DW_TAG_variable
	.long	15                              ; DW_AT_name
	.long	71                              ; DW_AT_type
                                        ; DW_AT_external
	.byte	1                               ; DW_AT_decl_file
	.byte	1                               ; DW_AT_decl_line
	.byte	6                               ; Abbrev [6] 0x47:0x8 DW_TAG_LLVM_ptrauth_type
	.long	59                              ; DW_AT_type
	.byte	4                               ; DW_AT_LLVM_ptrauth_key
                                        ; DW_AT_LLVM_ptrauth_address_discriminated
	.short	1235                            ; DW_AT_LLVM_ptrauth_extra_discriminator
                                        ; DW_AT_LLVM_ptrauth_isa_pointer
	.byte	5                               ; Abbrev [5] 0x4f:0xb DW_TAG_variable
	.long	18                              ; DW_AT_name
	.long	90                              ; DW_AT_type
                                        ; DW_AT_external
	.byte	1                               ; DW_AT_decl_file
	.byte	1                               ; DW_AT_decl_line
	.byte	7                               ; Abbrev [7] 0x5a:0x9 DW_TAG_LLVM_ptrauth_type
	.long	59                              ; DW_AT_type
	.byte	4                               ; DW_AT_LLVM_ptrauth_key
                                        ; DW_AT_LLVM_ptrauth_address_discriminated
	.short	1236                            ; DW_AT_LLVM_ptrauth_extra_discriminator
                                        ; DW_AT_LLVM_ptrauth_authenticates_null_values
	.byte	1                               ; DW_AT_LLVM_ptrauth_authentication_mode
	.byte	5                               ; Abbrev [5] 0x63:0xb DW_TAG_variable
	.long	21                              ; DW_AT_name
	.long	110                             ; DW_AT_type
                                        ; DW_AT_external
	.byte	1                               ; DW_AT_decl_file
	.byte	1                               ; DW_AT_decl_line
	.byte	8                               ; Abbrev [8] 0x6e:0x9 DW_TAG_LLVM_ptrauth_type
	.long	59                              ; DW_AT_type
	.byte	4                               ; DW_AT_LLVM_ptrauth_key
                                        ; DW_AT_LLVM_ptrauth_address_discriminated
	.short	1237                            ; DW_AT_LLVM_ptrauth_extra_discriminator
                                        ; DW_AT_LLVM_ptrauth_isa_pointer
                                        ; DW_AT_LLVM_ptrauth_authenticates_null_values
	.byte	2                               ; DW_AT_LLVM_ptrauth_authentication_mode
	.byte	0                               ; End Of Children Mark
Ldebug_info_end0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.byte	0                               ; string offset=0
	.asciz	"/tmp/p.c"                      ; string offset=1
	.asciz	"/"                             ; string offset=10
	.asciz	"p1"                            ; string offset=12
	.asciz	"p2"                            ; string offset=15
	.asciz	"p3"                            ; string offset=18
	.asciz	"p4"                            ; string offset=21
	.section	__DWARF,__apple_names,regular,debug
Lnames_begin:
	.long	1212240712                      ; Header Magic
	.short	1                               ; Header Version
	.short	0                               ; Header Hash Function
	.long	1                               ; Header Bucket Count
	.long	1                               ; Header Hash Count
	.long	12                              ; Header Data Length
	.long	0                               ; HeaderData Die Offset Base
	.long	1                               ; HeaderData Atom Count
	.short	1                               ; DW_ATOM_die_offset
	.short	6                               ; DW_FORM_data4
	.long	0                               ; Bucket 0
	.long	5863654                         ; Hash in Bucket 0
.set Lset3, LNames0-Lnames_begin        ; Offset in Bucket 0
	.long	Lset3
LNames0:
	.long	12                              ; p1
	.long	1                               ; Num DIEs
	.long	30
	.long	0
	.section	__DWARF,__apple_objc,regular,debug
Lobjc_begin:
	.long	1212240712                      ; Header Magic
	.short	1                               ; Header Version
	.short	0                               ; Header Hash Function
	.long	1                               ; Header Bucket Count
	.long	0                               ; Header Hash Count
	.long	12                              ; Header Data Length
	.long	0                               ; HeaderData Die Offset Base
	.long	1                               ; HeaderData Atom Count
	.short	1                               ; DW_ATOM_die_offset
	.short	6                               ; DW_FORM_data4
	.long	-1                              ; Bucket 0
	.section	__DWARF,__apple_namespac,regular,debug
Lnamespac_begin:
	.long	1212240712                      ; Header Magic
	.short	1                               ; Header Version
	.short	0                               ; Header Hash Function
	.long	1                               ; Header Bucket Count
	.long	0                               ; Header Hash Count
	.long	12                              ; Header Data Length
	.long	0                               ; HeaderData Die Offset Base
	.long	1                               ; HeaderData Atom Count
	.short	1                               ; DW_ATOM_die_offset
	.short	6                               ; DW_FORM_data4
	.long	-1                              ; Bucket 0
	.section	__DWARF,__apple_types,regular,debug
Ltypes_begin:
	.long	1212240712                      ; Header Magic
	.short	1                               ; Header Version
	.short	0                               ; Header Hash Function
	.long	1                               ; Header Bucket Count
	.long	0                               ; Header Hash Count
	.long	20                              ; Header Data Length
	.long	0                               ; HeaderData Die Offset Base
	.long	3                               ; HeaderData Atom Count
	.short	1                               ; DW_ATOM_die_offset
	.short	6                               ; DW_FORM_data4
	.short	3                               ; DW_ATOM_die_tag
	.short	5                               ; DW_FORM_data2
	.short	4                               ; DW_ATOM_type_flags
	.short	11                              ; DW_FORM_data1
	.long	-1                              ; Bucket 0
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
