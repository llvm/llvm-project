# RUN: llvm-mc %s -filetype obj -triple i686-pc-linux -o %t

## Check we don't crash when parsing invalid expression opcode.
# RUN: llvm-dwarfdump %t | FileCheck %s
# CHECK:      DW_TAG_GNU_call_site_parameter
# CHECK-NEXT:  DW_AT_location  (<decoding error> ff)

## Check verifier reports an error.
# RUN: not llvm-dwarfdump -verify %t 2>&1 | FileCheck %s --check-prefix=VERIFY
# VERIFY:      DIE contains invalid DWARF expression:
# VERIFY:      DW_TAG_GNU_call_site_parameter
# VERIFY-NEXT:   DW_AT_location    [DW_FORM_exprloc] (<decoding error> ff)

.section  .debug_info,"",@progbits
  .long  0x12
  .value  0x4
  .long  0
  .byte  0x4

  .uleb128 0x1 # DW_TAG_compile_unit [1]
  .long  0
  .byte  0x0

  .uleb128 0x2  # DW_TAG_GNU_call_site_parameter [2]
  .uleb128 0x1  # Expression size.
  .byte  0xff   # Broken expression.

  .byte  0      # End mark.
  .byte  0      # End mark.

.section .debug_abbrev,"",@progbits
  .uleb128 0x1    # ID [1]
  .uleb128 0x11   # DW_TAG_compile_unit, DW_CHILDREN_yes
  .byte  0x1
  .uleb128 0x25   # DW_AT_producer, DW_FORM_strp
  .uleb128 0xe
  .uleb128 0x13   # DW_AT_language, DW_FORM_data1
  .uleb128 0xb
  .byte  0
  .byte  0

  .uleb128 0x2    # ID [2]
  .uleb128 0x410a # DW_TAG_GNU_call_site_parameter, DW_CHILDREN_no
  .byte  0
  .uleb128 0x2    # DW_AT_location, DW_FORM_exprloc
  .uleb128 0x18
  .byte  0
  .byte  0
  .byte  0

.section .debug_str,"MS",@progbits,1
.string "test"
