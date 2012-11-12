// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

defined1:
defined2:
defined3:
        .symver defined1, bar1@zed
        .symver undefined1, bar2@zed

        .symver defined2, bar3@@zed

        .symver defined3, bar5@@@zed
        .symver undefined3, bar6@@@zed

        .long defined1
        .long undefined1
        .long defined2
        .long defined3
        .long undefined3

        .global global1
        .symver global1, g1@@zed
global1:


// CHECK:      # Relocation 0
// CHECK-NEXT: (('r_offset', 0x0000000000000000)
// CHECK-NEXT:  ('r_sym', 0x00000006)
// CHECK-NEXT:  ('r_type', 0x0000000a)
// CHECK-NEXT:  ('r_addend', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Relocation 1
// CHECK-NEXT: (('r_offset', 0x0000000000000004)
// CHECK-NEXT:  ('r_sym', 0x0000000b)
// CHECK-NEXT:  ('r_type', 0x0000000a)
// CHECK-NEXT:  ('r_addend', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Relocation 2
// CHECK-NEXT: (('r_offset', 0x0000000000000008)
// CHECK-NEXT:  ('r_sym', 0x00000006)
// CHECK-NEXT:  ('r_type', 0x0000000a)
// CHECK-NEXT:  ('r_addend', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Relocation 3
// CHECK-NEXT: (('r_offset', 0x000000000000000c)
// CHECK-NEXT:  ('r_sym', 0x00000006)
// CHECK-NEXT:  ('r_type', 0x0000000a)
// CHECK-NEXT:  ('r_addend', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Relocation 4
// CHECK-NEXT: (('r_offset', 0x0000000000000010)
// CHECK-NEXT:  ('r_sym', 0x0000000c)
// CHECK-NEXT:  ('r_type', 0x0000000a)
// CHECK-NEXT:  ('r_addend', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT:])

// CHECK:      # Symbol 1
// CHECK-NEXT: (('st_name', 0x00000013) # 'bar1@zed'
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 2
// CHECK-NEXT: (('st_name', 0x00000025) # 'bar3@@zed'
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 3
// CHECK-NEXT: (('st_name', 0x0000002f) # 'bar5@@zed'
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 4
// CHECK-NEXT: (('st_name', 0x00000001) # 'defined1'
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 5
// CHECK-NEXT: (('st_name', 0x0000000a) # 'defined2'
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 6
// CHECK-NEXT: (('st_name', 0x00000000) # ''
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x3)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 7
// CHECK-NEXT: (('st_name', 0x00000000) # ''
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x3)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0003)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 8
// CHECK-NEXT: (('st_name', 0x00000000) # ''
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x3)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0004)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 9
// CHECK-NEXT: (('st_name', 0x0000004a) # 'g1@@zed'
// CHECK-NEXT:  ('st_bind', 0x1)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0001)
// CHECK-NEXT:  ('st_value', 0x0000000000000014)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 10
// CHECK-NEXT: (('st_name', 0x00000042) # 'global1'
// CHECK-NEXT:  ('st_bind', 0x1)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0001)
// CHECK-NEXT:  ('st_value', 0x0000000000000014)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 11
// CHECK-NEXT: (('st_name', 0x0000001c) # 'bar2@zed'
// CHECK-NEXT:  ('st_bind', 0x1)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0000)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 12
// CHECK-NEXT: (('st_name', 0x00000039) # 'bar6@zed'
// CHECK-NEXT:  ('st_bind', 0x1)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0000)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT:])
