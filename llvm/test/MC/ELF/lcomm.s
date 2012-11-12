// RUN: llvm-mc -triple i386-pc-linux-gnu %s -filetype=obj -o - | elf-dump | FileCheck %s

.lcomm A, 5
.lcomm B, 32 << 20

// CHECK: (('st_name', 0x00000001) # 'A'
// CHECK:  ('st_value', 0x00000000)
// CHECK:  ('st_size', 0x00000005)
// CHECK:  ('st_bind', 0x0)
// CHECK:  ('st_type', 0x1)
// CHECK:  ('st_other', 0x00)
// CHECK:  ('st_shndx', 0x0003)
// CHECK: ),
// CHECK: (('st_name', 0x00000003) # 'B'
// CHECK:  ('st_value', 0x00000005)
// CHECK:  ('st_size', 0x02000000)
// CHECK:  ('st_bind', 0x0)
// CHECK:  ('st_type', 0x1)
// CHECK:  ('st_other', 0x00)
// CHECK:  ('st_shndx', 0x0003)
// CHECK: ),
