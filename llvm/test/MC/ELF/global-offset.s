// RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// We test that _GLOBAL_OFFSET_TABLE_ will account for the two bytes at the
// start of the addl/leal.

        addl    $_GLOBAL_OFFSET_TABLE_, %ebx
        leal    _GLOBAL_OFFSET_TABLE_(%ebx), %ebx

// But not in this case
foo:
        addl    _GLOBAL_OFFSET_TABLE_-foo,%ebx

// CHECK:      ('sh_name', 0x00000005) # '.text'
// CHECK-NEXT: ('sh_type',
// CHECK-NEXT: ('sh_flags',
// CHECK-NEXT: ('sh_addr',
// CHECK-NEXT: ('sh_offset',
// CHECK-NEXT: ('sh_size',
// CHECK-NEXT: ('sh_link',
// CHECK-NEXT: ('sh_info',
// CHECK-NEXT: ('sh_addralign',
// CHECK-NEXT: ('sh_entsize',
// CHECK-NEXT: ('_section_data', '81c30200 00008d9b 02000000 031d0200 0000')
