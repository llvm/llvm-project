// This checks that llvm-mc is able to produce 64-bit debug info.

// RUN: llvm-mc -g -dwarf64 -triple x86_64-apple-darwin10 %s -filetype=obj -o %t
// RUN: llvm-dwarfdump -v %t | FileCheck %s

// CHECK:      .debug_info contents:
// CHECK-NEXT: 0x00000000: Compile Unit: {{.*}} format = DWARF64, version = {{.*}}, abbr_offset = 0x0000, addr_size = 0x08
// CHECK:      DW_TAG_compile_unit [1] *
// CHECK-NEXT:   DW_AT_stmt_list [DW_FORM_sec_offset] (0x0000000000000000)
// CHECK:      DW_TAG_label [2]
// CHECK-NEXT:   DW_AT_name [DW_FORM_string] ("foo")

// CHECK:      .debug_aranges contents:
// CHECK-NEXT: Address Range Header: length = 0x0000000000000034, format = DWARF64, version = 0x0002, cu_offset = 0x0000000000000000, addr_size = 0x08, seg_size = 0x00
// CHECK-NEXT: [0x0000000000000000,  0x0000000000000001)
// CHECK-EMPTY:

_foo:
    nop
