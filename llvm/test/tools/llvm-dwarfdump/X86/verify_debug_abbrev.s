# RUN: llvm-mc %s -filetype obj -triple x86_64-unknown-linux-gnu -o - \
# RUN: | not llvm-dwarfdump -verify - \
# RUN: | FileCheck %s

# CHECK: Verifying .debug_abbrev...
# CHECK-NEXT: error: Abbreviation declaration contains multiple DW_AT_stmt_list attributes.
# CHECK-NEXT:[1] DW_TAG_compile_unit	DW_CHILDREN_no
# CHECK-NEXT:	DW_AT_stmt_list	DW_FORM_sec_offset
# CHECK-NEXT:	DW_AT_GNU_dwo_name	DW_FORM_strp
# CHECK-NEXT:	DW_AT_stmt_list	DW_FORM_strp{{[[:space:]]}}
# CHECK-NEXT: error: Abbreviation declaration contains multiple DW_AT_producer attributes.
# CHECK-NEXT:[1] DW_TAG_compile_unit	DW_CHILDREN_yes
# CHECK-NEXT:	DW_AT_GNU_dwo_name	DW_FORM_GNU_str_index
# CHECK-NEXT:	DW_AT_producer	DW_FORM_GNU_str_index
# CHECK-NEXT:	DW_AT_producer	DW_FORM_data2


  .section  .debug_abbrev,"",@progbits
  .byte  1                       # Abbreviation Code
  .byte  17                      # DW_TAG_compile_unit
  .byte  0                       # DW_CHILDREN_no
  .byte  16                      # DW_AT_stmt_list
  .byte  23                      # DW_FORM_sec_offset
  .ascii  "\260B"                # DW_AT_GNU_dwo_name
  .byte  14                      # DW_FORM_strp
  .byte  16                      # DW_AT_stmt_list -- Error: Abbreviation declaration contains multiple DW_AT_stmt_list attributes.
  .byte  14                      # DW_FORM_strp
  .byte  0                       # EOM(1)
  .byte  0                       # EOM(2)
  .byte  0                       # EOM(3)
  .section  .debug_abbrev.dwo,"",@progbits
  .byte  1                       # Abbreviation Code
  .byte  17                      # DW_TAG_compile_unit
  .byte  1                       # DW_CHILDREN_yes
  .ascii  "\260B"                # DW_AT_GNU_dwo_name
  .ascii  "\202>"                # DW_FORM_GNU_str_index
  .byte  37                      # DW_AT_producer
  .ascii  "\202>"                # DW_FORM_GNU_str_index
  .byte  37                      # DW_AT_producer -- Error: Abbreviation declaration contains multiple DW_AT_producer attributes.
  .byte  5                       # DW_FORM_data1
  .byte  0                       # EOM(1)
  .byte  0                       # EOM(2)
  .byte  0                       # EOM(3)
