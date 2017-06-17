# RUN: llvm-mc %s -filetype obj -triple x86_64-apple-darwin -o - \
# RUN: | not llvm-dwarfdump -verify - \
# RUN: | FileCheck %s

# CHECK-NOT: Verifying .apple_names...

# This test is meant to verify that the -verify option 
# in llvm-dwarfdump doesn't produce any .apple_names related
# output when there's no such section int he object.
# The test was manually modified to exclude the 
# .apple_names section from the apple_names_verify_buckets.s
# test file in the same directory.

  .section  __TEXT,__text,regular,pure_instructions
  .file 1 "basic.c"
  .comm _i,4,2                  ## @i
  .section  __DWARF,__debug_str,regular,debug
Linfo_string:
  .asciz  "basic.c"               ## string offset=42
  .asciz  "i"                     ## string offset=84
  .asciz  "int"                   ## string offset=86
  .section  __DWARF,__debug_loc,regular,debug
Lsection_debug_loc:
  .section  __DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
  .byte 1                       ## Abbreviation Code
  .byte 17                      ## DW_TAG_compile_unit
  .byte 1                       ## DW_CHILDREN_yes
  .byte 37                      ## DW_AT_producer
  .byte 14                      ## DW_FORM_strp
  .byte 19                      ## DW_AT_language
  .byte 5                       ## DW_FORM_data2
  .byte 3                       ## DW_AT_name
  .byte 14                      ## DW_FORM_strp
  .byte 16                      ## DW_AT_stmt_list
  .byte 23                      ## DW_FORM_sec_offset
  .byte 27                      ## DW_AT_comp_dir
  .byte 14                      ## DW_FORM_strp
  .byte 0                       ## EOM(1)
  .byte 0                       ## EOM(2)
  .byte 2                       ## Abbreviation Code
  .byte 52                      ## DW_TAG_variable
  .byte 0                       ## DW_CHILDREN_no
  .byte 3                       ## DW_AT_name
  .byte 14                      ## DW_FORM_strp
  .byte 73                      ## DW_AT_type
  .byte 19                      ## DW_FORM_ref4
  .byte 63                      ## DW_AT_external
  .byte 25                      ## DW_FORM_flag_present
  .byte 58                      ## DW_AT_decl_file
  .byte 11                      ## DW_FORM_data1
  .byte 59                      ## DW_AT_decl_line
  .byte 11                      ## DW_FORM_data1
  .byte 2                       ## DW_AT_location
  .byte 24                      ## DW_FORM_exprloc
  .byte 0                       ## EOM(1)
  .byte 0                       ## EOM(2)
  .byte 3                       ## Abbreviation Code
  .byte 36                      ## DW_TAG_base_type
  .byte 0                       ## DW_CHILDREN_no
  .byte 3                       ## DW_AT_name
  .byte 14                      ## DW_FORM_strp
  .byte 62                      ## DW_AT_encoding
  .byte 11                      ## DW_FORM_data1
  .byte 11                      ## DW_AT_byte_size
  .byte 11                      ## DW_FORM_data1
  .byte 0                       ## EOM(1)
  .byte 0                       ## EOM(2)
  .byte 0                       ## EOM(3)
  .section  __DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
  .long 55                      ## Length of Unit
  .short  4                       ## DWARF version number
Lset0 = Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
  .long Lset0
  .byte 8                       ## Address Size (in bytes)
  .byte 1                       ## Abbrev [1] 0xb:0x30 DW_TAG_compile_unit
  .long 0                       ## DW_AT_producer
  .short  12                      ## DW_AT_language
  .long 42                      ## DW_AT_name
Lset1 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
  .long Lset1
  .long 50                      ## DW_AT_comp_dir
  .byte 2                       ## Abbrev [2] 0x1e:0x15 DW_TAG_variable
  .long 84                      ## DW_AT_name
  .long 51                      ## DW_AT_type
                                        ## DW_AT_external
  .byte 1                       ## DW_AT_decl_file
  .byte 1                       ## DW_AT_decl_line
  .byte 9                       ## DW_AT_location
  .byte 3
  .quad _i
  .byte 3                       ## Abbrev [3] 0x33:0x7 DW_TAG_base_type
  .long 86                      ## DW_AT_name
  .byte 5                       ## DW_AT_encoding
  .byte 4                       ## DW_AT_byte_size
  .byte 0                       ## End Of Children Mark
  .section  __DWARF,__debug_ranges,regular,debug
Ldebug_range:
  .section  __DWARF,__debug_macinfo,regular,debug
Ldebug_macinfo:
Lcu_macro_begin0:
  .byte 0                       ## End Of Macro List Mark
  .section  __DWARF,__apple_objc,regular,debug
Lobjc_begin:
  .long 1212240712              ## Header Magic
  .short  1                       ## Header Version
  .short  0                       ## Header Hash Function
  .long 1                       ## Header Bucket Count
  .long 0                       ## Header Hash Count
  .long 12                      ## Header Data Length
  .long 0                       ## HeaderData Die Offset Base
  .long 1                       ## HeaderData Atom Count
  .short  1                       ## DW_ATOM_die_offset
  .short  6                       ## DW_FORM_data4
  .long -1                      ## Bucket 0
  .section  __DWARF,__apple_namespac,regular,debug
Lnamespac_begin:
  .long 1212240712              ## Header Magic
  .short  1                       ## Header Version
  .short  0                       ## Header Hash Function
  .long 1                       ## Header Bucket Count
  .long 0                       ## Header Hash Count
  .long 12                      ## Header Data Length
  .long 0                       ## HeaderData Die Offset Base
  .long 1                       ## HeaderData Atom Count
  .short  1                       ## DW_ATOM_die_offset
  .short  6                       ## DW_FORM_data4
  .long -1                      ## Bucket 0
  .section  __DWARF,__apple_types,regular,debug
Ltypes_begin:
  .long 1212240712              ## Header Magic
  .short  1                       ## Header Version
  .short  0                       ## Header Hash Function
  .long 1                       ## Header Bucket Count
  .long 1                       ## Header Hash Count
  .long 20                      ## Header Data Length
  .long 0                       ## HeaderData Die Offset Base
  .long 3                       ## HeaderData Atom Count
  .short  1                       ## DW_ATOM_die_offset
  .short  6                       ## DW_FORM_data4
  .short  3                       ## DW_ATOM_die_tag
  .short  5                       ## DW_FORM_data2
  .short  4                       ## DW_ATOM_type_flags
  .short  11                      ## DW_FORM_data1
  .long 0                       ## Bucket 0
  .long 193495088               ## Hash in Bucket 0
  .long Ltypes0-Ltypes_begin    ## Offset in Bucket 0
Ltypes0:
  .long 86                      ## int
  .long 1                       ## Num DIEs
  .long 51
  .short  36
  .byte 0
  .long 0
  .section  __DWARF,__apple_exttypes,regular,debug
Lexttypes_begin:
  .long 1212240712              ## Header Magic
  .short  1                       ## Header Version
  .short  0                       ## Header Hash Function
  .long 1                       ## Header Bucket Count
  .long 0                       ## Header Hash Count
  .long 12                      ## Header Data Length
  .long 0                       ## HeaderData Die Offset Base
  .long 1                       ## HeaderData Atom Count
  .short  7                       ## DW_ATOM_ext_types
  .short  6                       ## DW_FORM_data4
  .long -1                      ## Bucket 0

.subsections_via_symbols
  .section  __DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
