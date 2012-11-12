// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that both % and @ are accepted.
        .global foo
        .type foo,%function
foo:

        .global bar
        .type bar,@object
bar:

// Test that gnu_unique_object is accepted.
        .type zed,@gnu_unique_object

ifunc:
        .global ifunc
        .type ifunc,@gnu_indirect_function

// CHECK:      # Symbol 4
// CHECK-NEXT: (('st_name', 0x00000005) # 'bar'
// CHECK-NEXT:  ('st_bind', 0x1)
// CHECK-NEXT:  ('st_type', 0x1)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 5
// CHECK-NEXT: (('st_name', 0x00000001) # 'foo'
// CHECK-NEXT:  ('st_bind', 0x1)
// CHECK-NEXT:  ('st_type', 0x2)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 6
// CHECK-NEXT: (('st_name', 0x00000009) # 'ifunc'
// CHECK-NEXT:  ('st_bind', 0x1)
// CHECK-NEXT:  ('st_type', 0xa)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),

