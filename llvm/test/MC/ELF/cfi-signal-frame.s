// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck %s

f:
        .cfi_startproc
        .cfi_signal_frame
        .cfi_endproc

g:
        .cfi_startproc
        .cfi_endproc

// CHECK:      (('sh_name', 0x00000011) # '.eh_frame'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x0000000000000002)
// CHECK-NEXT:  ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:  ('sh_offset', 0x0000000000000040)
// CHECK-NEXT:  ('sh_size', 0x0000000000000058)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x0000000000000008)
// CHECK-NEXT:  ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT:  ('_section_data', '14000000 00000000 017a5253 00017810 011b0c07 08900100 10000000 1c000000 00000000 00000000 00000000 14000000 00000000 017a5200 01781001 1b0c0708 90010000 10000000 1c000000 00000000 00000000 00000000')
// CHECK-NEXT: ),
