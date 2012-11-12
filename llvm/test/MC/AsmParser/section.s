# RUN: llvm-mc -triple i386-pc-linux-gnu -filetype=obj -o %t %s
# RUN: elf-dump --dump-section-data < %t | FileCheck %s
.section test1
.byte 1
.section test2
.byte 2
.previous
.byte 1
.section test2
.byte 2
.previous
.byte 1
.section test1
.byte 1
.previous
.byte 1
.section test2
.byte 2
.pushsection test3
.byte 3
.pushsection test4
.byte 4
.pushsection test5
.byte 5
.popsection
.byte 4
.popsection
.byte 3
.popsection
.byte 2
.pushsection test3
.byte 3
.pushsection test4
.byte 4
.previous
.byte 3
.popsection
.byte 3
.previous
.byte 2
.section test1
.byte 1
.popsection
.byte 2
.previous
.byte 1
.previous
# CHECK:       (('sh_name', 0x00000044) # 'test1'
# CHECK-NEXT:   ('sh_type', 0x00000001)
# CHECK-NEXT:   ('sh_flags', 0x00000000)
# CHECK-NEXT:   ('sh_addr', 0x00000000)
# CHECK-NEXT:   ('sh_offset', 0x00000034)
# CHECK-NEXT:   ('sh_size', 0x00000007)
# CHECK-NEXT:   ('sh_link', 0x00000000)
# CHECK-NEXT:   ('sh_info', 0x00000000)
# CHECK-NEXT:   ('sh_addralign', 0x00000001)
# CHECK-NEXT:   ('sh_entsize', 0x00000000)
# CHECK-NEXT:   ('_section_data', '01010101 010101')
# CHECK-NEXT:  ),
# CHECK:       (('sh_name', 0x0000003e) # 'test2'
# CHECK-NEXT:   ('sh_type', 0x00000001)
# CHECK-NEXT:   ('sh_flags', 0x00000000)
# CHECK-NEXT:   ('sh_addr', 0x00000000)
# CHECK-NEXT:   ('sh_offset', 0x0000003b)
# CHECK-NEXT:   ('sh_size', 0x00000006)
# CHECK-NEXT:   ('sh_link', 0x00000000)
# CHECK-NEXT:   ('sh_info', 0x00000000)
# CHECK-NEXT:   ('sh_addralign', 0x00000001)
# CHECK-NEXT:   ('sh_entsize', 0x00000000)
# CHECK-NEXT:   ('_section_data', '02020202 0202')
# CHECK-NEXT:  ),
# CHECK:       (('sh_name', 0x00000038) # 'test3'
# CHECK-NEXT:   ('sh_type', 0x00000001)
# CHECK-NEXT:   ('sh_flags', 0x00000000)
# CHECK-NEXT:   ('sh_addr', 0x00000000)
# CHECK-NEXT:   ('sh_offset', 0x00000041)
# CHECK-NEXT:   ('sh_size', 0x00000005)
# CHECK-NEXT:   ('sh_link', 0x00000000)
# CHECK-NEXT:   ('sh_info', 0x00000000)
# CHECK-NEXT:   ('sh_addralign', 0x00000001)
# CHECK-NEXT:   ('sh_entsize', 0x00000000)
# CHECK-NEXT:   ('_section_data', '03030303 03')
# CHECK-NEXT:  ),
# CHECK:       (('sh_name', 0x00000032) # 'test4'
# CHECK-NEXT:   ('sh_type', 0x00000001)
# CHECK-NEXT:   ('sh_flags', 0x00000000)
# CHECK-NEXT:   ('sh_addr', 0x00000000)
# CHECK-NEXT:   ('sh_offset', 0x00000046)
# CHECK-NEXT:   ('sh_size', 0x00000003)
# CHECK-NEXT:   ('sh_link', 0x00000000)
# CHECK-NEXT:   ('sh_info', 0x00000000)
# CHECK-NEXT:   ('sh_addralign', 0x00000001)
# CHECK-NEXT:   ('sh_entsize', 0x00000000)
# CHECK-NEXT:   ('_section_data', '040404')
# CHECK-NEXT:  ),
# CHECK:       (('sh_name', 0x0000002c) # 'test5'
# CHECK-NEXT:   ('sh_type', 0x00000001)
# CHECK-NEXT:   ('sh_flags', 0x00000000)
# CHECK-NEXT:   ('sh_addr', 0x00000000)
# CHECK-NEXT:   ('sh_offset', 0x00000049)
# CHECK-NEXT:   ('sh_size', 0x00000001)
# CHECK-NEXT:   ('sh_link', 0x00000000)
# CHECK-NEXT:   ('sh_info', 0x00000000)
# CHECK-NEXT:   ('sh_addralign', 0x00000001)
# CHECK-NEXT:   ('sh_entsize', 0x00000000)
# CHECK-NEXT:   ('_section_data', '05')
# CHECK-NEXT:  ),
