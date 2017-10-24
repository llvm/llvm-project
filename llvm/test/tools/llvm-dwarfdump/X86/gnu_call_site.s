# RUN: llvm-mc %s -filetype obj -triple i686-pc-linux -o %t
# RUN: llvm-dwarfdump %t | FileCheck %s

# CHECK:      DW_TAG_subprogram
# CHECK-NEXT:   DW_AT_external  (true)
# CHECK-NEXT:   DW_AT_name      ("fn4")
# CHECK-NEXT:   DW_AT_linkage_name      ()
# CHECK-NEXT:   DW_AT_low_pc    (0x0000000000000000)
# CHECK-NEXT:   DW_AT_high_pc   (0x00000000)
# CHECK-NEXT:   DW_AT_frame_base        (DW_OP_call_frame_cfa)
# CHECK-NEXT:   DW_AT_GNU_all_call_sites        (true)

# CHECK:      DW_TAG_GNU_call_site
# CHECK-NEXT:   DW_AT_low_pc  (0x0000000000000000)
# CHECK-NEXT:   DW_AT_abstract_origin (cu + 0x0001)

# CHECK:      DW_TAG_GNU_call_site_parameter
# CHECK-NEXT:   DW_AT_location      (DW_OP_reg0 EAX)
# CHECK-NEXT:   DW_AT_GNU_call_site_value   (DW_OP_addr 0x0)

.section  .debug_info,"",@progbits
  .long  0x47
  .value  0x4
  .long  0
  .byte  0x4

  .uleb128 0x1 # DW_TAG_compile_unit [1]
  .long  0
  .byte  0x0
  .long  0
  .long  0
  .long  0
  .long  0
  .long  0

  .uleb128 0xe # DW_TAG_subprogram [14]
  .string  "fn4"
  .long  0
  .long  0
  .long  0
  .uleb128 0x1  # DW_AT_GNU_all_call_sites
  .byte  0x9c

  .uleb128 0x12 # DW_TAG_GNU_call_site [18]
  .long  0x0
  .long  0x1

  .uleb128 0x13 # DW_TAG_GNU_call_site_parameter [19]
  .uleb128 0x1
  .byte  0x50
  .uleb128 0x5
  .byte  0x3
  .long  X
  .byte  0
  .byte  0
  .byte  0

.section .debug_abbrev,"",@progbits
  .uleb128 0x1
  .uleb128 0x11
  .byte  0x1 # [1]
  .uleb128 0x25
  .uleb128 0xe
  .uleb128 0x13
  .uleb128 0xb
  .uleb128 0x3
  .uleb128 0xe
  .uleb128 0x1b
  .uleb128 0xe
  .uleb128 0x11
  .uleb128 0x1
  .uleb128 0x12
  .uleb128 0x6
  .uleb128 0x10
  .uleb128 0x17
  .byte  0
  .byte  0

  .uleb128 0xe # [14]
  .uleb128 0x2e
  .byte  0x1
  .uleb128 0x3f
  .uleb128 0x19
  .uleb128 0x3
  .uleb128 0x8
  .uleb128 0x6e
  .uleb128 0xe
  .uleb128 0x11
  .uleb128 0x1
  .uleb128 0x12
  .uleb128 0x6
  .uleb128 0x40
  .uleb128 0x18
  .uleb128 0x2117
  .uleb128 0x19
  .byte  0
  .byte  0

  .uleb128 0x12 # [18]
  .uleb128 0x4109
  .byte  0x1
  .uleb128 0x11
  .uleb128 0x1
  .uleb128 0x31
  .uleb128 0x13
  .byte  0
  .byte  0

  .uleb128 0x13 # [19]
  .uleb128 0x410a
  .byte  0
  .uleb128 0x2
  .uleb128 0x18
  .uleb128 0x2111
  .uleb128 0x18
  .byte  0
  .byte  0
  .byte  0
