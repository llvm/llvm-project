# RUN: llvm-mc %s -filetype obj -triple=i686-pc-linux -o %t
# RUN: not llvm-dwarfdump -v -verify %t 2>&1 | FileCheck %s
# CHECK: The length for this unit is too large for the .debug_info provided.

## Check we do not crash when trying to parse truncated .debug_info.
.section  .debug_info,"",@progbits
  .long 0x1c
  .value  0x4
  .long  .Ldebug_abbrev0
  .byte  0x4

  .uleb128 0x1 # DW_TAG_compile_unit [1] *
  .long  0     # DW_AT_producer [DW_FORM_strp] ( .debug_str[0x00000000] = "test")
  .byte  0x4   # DW_AT_language [DW_FORM_data1] (DW_LANG_C_plus_plus)
  .long  0     # DW_AT_name [DW_FORM_strp] ( .debug_str[0x00000000] = "test")
  .long  0     # DW_AT_comp_dir [DW_FORM_strp] ( .debug_str[0x00000000] = "test")
  .long  0     # DW_AT_low_pc [DW_FORM_addr] (0x0000000000000000)
  .long  0     # DW_AT_high_pc [DW_FORM_data4] (0x00000000)

.section  .debug_abbrev,"",@progbits
.Ldebug_abbrev0:
  .uleb128 0x1
  .uleb128 0x11 # DW_TAG_compile_unit, DW_CHILDREN_yes
  .byte  0x1
  .uleb128 0x25 # DW_AT_producer, DW_FORM_strp
  .uleb128 0xe
  .uleb128 0x13 # DW_AT_language, DW_FORM_data1
  .uleb128 0xb
  .uleb128 0x3  # DW_AT_name, DW_FORM_strp
  .uleb128 0xe
  .uleb128 0x1b # DW_AT_comp_dir, DW_FORM_strp
  .uleb128 0xe
  .uleb128 0x11 # DW_AT_low_pc, DW_FORM_addr
  .uleb128 0x1
  .uleb128 0x12 # DW_AT_high_pc, DW_FORM_data4
  .uleb128 0x6
  .byte  0
  .byte  0
  .byte  0

.section .debug_str,"MS",@progbits,1
.string "test"
