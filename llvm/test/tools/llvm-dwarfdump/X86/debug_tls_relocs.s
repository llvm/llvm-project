# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump -v %t.o | FileCheck %s

# CHECK-NOT: error:
# CHECK: DW_AT_location [DW_FORM_exprloc] (DW_OP_const8u 0x0, DW_OP_GNU_push_tls_address)
# CHECK: DW_AT_location [DW_FORM_exprloc] (DW_OP_const4u 0x0, DW_OP_GNU_push_tls_address)

.section .debug_str,"MS",@progbits,1
.Linfo_string0:
 .asciz "X"

.section .debug_abbrev,"",@progbits
 .byte 1                # Abbreviation Code
 .byte 17               # DW_TAG_compile_unit
 .byte 1                # DW_CHILDREN_yes
 .byte 37               # DW_AT_producer
 .byte 14               # DW_FORM_strp
 .byte 19               # DW_AT_language
 .byte 5                # DW_FORM_data2
 .byte 3                # DW_AT_name
 .byte 14               # DW_FORM_strp
 .byte 0                # EOM(1)
 .byte 0                # EOM(2)

 .byte 2                # Abbreviation Code
 .byte 52               # DW_TAG_variable
 .byte 0                # DW_CHILDREN_no
 .byte 3                # DW_AT_name
 .byte 14               # DW_FORM_strp
 .byte 73               # DW_AT_type
 .byte 19               # DW_FORM_ref4
 .byte 63               # DW_AT_external
 .byte 25               # DW_FORM_flag_present
 .byte 2                # DW_AT_location
 .byte 24               # DW_FORM_exprloc
 .byte 0                # EOM(1)
 .byte 0                # EOM(2)

 .byte 0                # EOM(3)

.section .debug_info,"",@progbits
 .long 49               # Length of Unit
 .short 4               # DWARF version number
 .long .debug_abbrev    # Offset Into Abbrev. Section
 .byte 8                # Address Size (in bytes)
 .byte 1                # Abbrev [1] 0xb:0x6c DW_TAG_compile_unit
 .long .Linfo_string0   # DW_AT_producer
 .short 4               # DW_AT_language
 .long .Linfo_string0   # DW_AT_name
                        
 .byte 2                # Abbrev [2] 0x2a:0x16 DW_TAG_variable
 .long .Linfo_string0   # DW_AT_name
 .long 0                # DW_AT_type
 .byte 10               # DW_AT_location
 .byte 14
 .quad tdata1@DTPOFF
 .byte 224

 .byte 2                # Abbrev [2] 0x47:0x16 DW_TAG_variable
 .long .Linfo_string0   # DW_AT_name
 .long 0                # DW_AT_type
 .byte 6                # DW_AT_location
 .byte 12
 .long tdata2@DTPOFF
 .byte 224

 .byte 0                # End Of Children Mark
 .byte 0                # End Of Children Mark
