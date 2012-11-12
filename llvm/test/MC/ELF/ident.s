// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck %s

// CHECK:       (('sh_name', 0x00000007) # '.comment'
// CHECK-NEXT:   ('sh_type', 0x00000001)
// CHECK-NEXT:   ('sh_flags', 0x0000000000000030)
// CHECK-NEXT:   ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:   ('sh_offset', 0x0000000000000040)
// CHECK-NEXT:   ('sh_size', 0x000000000000000d)
// CHECK-NEXT:   ('sh_link', 0x00000000)
// CHECK-NEXT:   ('sh_info', 0x00000000)
// CHECK-NEXT:   ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:   ('sh_entsize', 0x0000000000000001)
// CHECK-NEXT:   ('_section_data', '00666f6f 00626172 007a6564 00')

        .ident "foo"
        .ident "bar"
        .ident "zed"
