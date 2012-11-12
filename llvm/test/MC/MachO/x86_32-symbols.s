// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck %s

        .text
L0:
D0:
        .section	__TEXT,__text,regular,pure_instructions
L1:
D1:
        .const
L2:
D2:
        .static_const
L3:
D3:
        .cstring
L4:
D4:
        .literal4
L5:
D5:
        .literal8
L6:
D6:
        .literal16
L7:
D7:
        .constructor
L8:
D8:
        .destructor
L9:
D9:
        .symbol_stub
L10:
D10:
        .picsymbol_stub
L11:
D11:
        .data
L12:
D12:
        .static_data
L13:
D13:
        .non_lazy_symbol_pointer
L14:
D14:
        .lazy_symbol_pointer
L15:
D15:
        .dyld
L16:
D16:
        .mod_init_func
L17:
D17:
        .mod_term_func
L18:
D18:
        .const_data
L19:
D19:
        .objc_class
L20:
D20:
        .objc_meta_class
L21:
D21:
        .objc_cat_cls_meth
L22:
D22:
        .objc_cat_inst_meth
L23:
D23:
        .objc_protocol
L24:
D24:
        .objc_string_object
L25:
D25:
        .objc_cls_meth
L26:
D26:
        .objc_inst_meth
L27:
D27:
        .objc_cls_refs
L28:
D28:
        .objc_message_refs
L29:
D29:
        .objc_symbols
L30:
D30:
        .objc_category
L31:
D31:
        .objc_class_vars
L32:
D32:
        .objc_instance_vars
L33:
D33:
        .objc_module_info
L34:
D34:
        .objc_class_names
L35:
D35:
        .objc_meth_var_types
L36:
D36:
        .objc_meth_var_names
L37:
D37:
        .objc_selector_strs
L38:
D38:
        .section __TEXT,__picsymbolstub4,symbol_stubs,none,16
L39:
D39:

// CHECK: ('cputype', 7)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 3)
// CHECK: ('load_commands_size', 2608)
// CHECK: ('flag', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 1)
// CHECK:   ('size', 2504)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 0)
// CHECK:   ('file_offset', 2636)
// CHECK:   ('file_size', 0)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 36)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 1
// CHECK:    (('section_name', '__const\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 2
// CHECK:    (('section_name', '__static_const\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 3
// CHECK:    (('section_name', '__cstring\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x2)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 4
// CHECK:    (('section_name', '__literal4\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 2)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x3)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 5
// CHECK:    (('section_name', '__literal8\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 3)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x4)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 6
// CHECK:    (('section_name', '__literal16\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 4)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0xe)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 7
// CHECK:    (('section_name', '__constructor\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 8
// CHECK:    (('section_name', '__destructor\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 9
// CHECK:    (('section_name', '__symbol_stub\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000008)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 16)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 10
// CHECK:    (('section_name', '__picsymbol_stub')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000008)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 26)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 11
// CHECK:    (('section_name', '__data\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 12
// CHECK:    (('section_name', '__static_data\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 13
// CHECK:    (('section_name', '__nl_symbol_ptr\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 2)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x6)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 14
// CHECK:    (('section_name', '__la_symbol_ptr\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 2)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x7)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 15
// CHECK:    (('section_name', '__dyld\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 16
// CHECK:    (('section_name', '__mod_init_func\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 2)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x9)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 17
// CHECK:    (('section_name', '__mod_term_func\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 2)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0xa)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 18
// CHECK:    (('section_name', '__const\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 19
// CHECK:    (('section_name', '__class\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 20
// CHECK:    (('section_name', '__meta_class\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 21
// CHECK:    (('section_name', '__cat_cls_meth\x00\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 22
// CHECK:    (('section_name', '__cat_inst_meth\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 23
// CHECK:    (('section_name', '__protocol\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 24
// CHECK:    (('section_name', '__string_object\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 25
// CHECK:    (('section_name', '__cls_meth\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 26
// CHECK:    (('section_name', '__inst_meth\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 27
// CHECK:    (('section_name', '__cls_refs\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 2)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000005)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 28
// CHECK:    (('section_name', '__message_refs\x00\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 2)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000005)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 29
// CHECK:    (('section_name', '__symbols\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 30
// CHECK:    (('section_name', '__category\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 31
// CHECK:    (('section_name', '__class_vars\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 32
// CHECK:    (('section_name', '__instance_vars\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 33
// CHECK:    (('section_name', '__module_info\x00\x00\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x10000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 34
// CHECK:    (('section_name', '__selector_strs\x00')
// CHECK:     ('segment_name', '__OBJC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x2)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 35
// CHECK:    (('section_name', '__picsymbolstub4')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 2636)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x8)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 16)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 2636)
// CHECK:   ('nsyms', 40)
// CHECK:   ('stroff', 3116)
// CHECK:   ('strsize', 152)
// CHECK:   ('_string_data', '\x00D0\x00D1\x00D2\x00D3\x00D4\x00D5\x00D6\x00D7\x00D8\x00D9\x00D10\x00D11\x00D12\x00D13\x00D14\x00D15\x00D16\x00D17\x00D18\x00D19\x00D20\x00D21\x00D22\x00D23\x00D24\x00D25\x00D26\x00D27\x00D28\x00D29\x00D30\x00D31\x00D32\x00D33\x00D34\x00D35\x00D36\x00D37\x00D38\x00D39\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D0')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 4)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D1')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 7)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D2')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 10)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 3)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D3')
// CHECK:    ),
// CHECK:     # Symbol 4
// CHECK:    (('n_strx', 13)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 4)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D4')
// CHECK:    ),
// CHECK:     # Symbol 5
// CHECK:    (('n_strx', 16)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 5)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D5')
// CHECK:    ),
// CHECK:     # Symbol 6
// CHECK:    (('n_strx', 19)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 6)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D6')
// CHECK:    ),
// CHECK:     # Symbol 7
// CHECK:    (('n_strx', 22)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 7)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D7')
// CHECK:    ),
// CHECK:     # Symbol 8
// CHECK:    (('n_strx', 25)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 8)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D8')
// CHECK:    ),
// CHECK:     # Symbol 9
// CHECK:    (('n_strx', 28)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 9)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D9')
// CHECK:    ),
// CHECK:     # Symbol 10
// CHECK:    (('n_strx', 31)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 10)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D10')
// CHECK:    ),
// CHECK:     # Symbol 11
// CHECK:    (('n_strx', 35)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 11)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D11')
// CHECK:    ),
// CHECK:     # Symbol 12
// CHECK:    (('n_strx', 39)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 12)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D12')
// CHECK:    ),
// CHECK:     # Symbol 13
// CHECK:    (('n_strx', 43)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 13)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D13')
// CHECK:    ),
// CHECK:     # Symbol 14
// CHECK:    (('n_strx', 47)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 14)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D14')
// CHECK:    ),
// CHECK:     # Symbol 15
// CHECK:    (('n_strx', 51)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 15)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D15')
// CHECK:    ),
// CHECK:     # Symbol 16
// CHECK:    (('n_strx', 55)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 16)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D16')
// CHECK:    ),
// CHECK:     # Symbol 17
// CHECK:    (('n_strx', 59)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 17)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D17')
// CHECK:    ),
// CHECK:     # Symbol 18
// CHECK:    (('n_strx', 63)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 18)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D18')
// CHECK:    ),
// CHECK:     # Symbol 19
// CHECK:    (('n_strx', 67)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 19)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D19')
// CHECK:    ),
// CHECK:     # Symbol 20
// CHECK:    (('n_strx', 71)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 20)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D20')
// CHECK:    ),
// CHECK:     # Symbol 21
// CHECK:    (('n_strx', 75)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 21)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D21')
// CHECK:    ),
// CHECK:     # Symbol 22
// CHECK:    (('n_strx', 79)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 22)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D22')
// CHECK:    ),
// CHECK:     # Symbol 23
// CHECK:    (('n_strx', 83)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 23)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D23')
// CHECK:    ),
// CHECK:     # Symbol 24
// CHECK:    (('n_strx', 87)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 24)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D24')
// CHECK:    ),
// CHECK:     # Symbol 25
// CHECK:    (('n_strx', 91)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 25)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D25')
// CHECK:    ),
// CHECK:     # Symbol 26
// CHECK:    (('n_strx', 95)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 26)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D26')
// CHECK:    ),
// CHECK:     # Symbol 27
// CHECK:    (('n_strx', 99)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 27)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D27')
// CHECK:    ),
// CHECK:     # Symbol 28
// CHECK:    (('n_strx', 103)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 28)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D28')
// CHECK:    ),
// CHECK:     # Symbol 29
// CHECK:    (('n_strx', 107)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 29)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D29')
// CHECK:    ),
// CHECK:     # Symbol 30
// CHECK:    (('n_strx', 111)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 30)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D30')
// CHECK:    ),
// CHECK:     # Symbol 31
// CHECK:    (('n_strx', 115)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 31)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D31')
// CHECK:    ),
// CHECK:     # Symbol 32
// CHECK:    (('n_strx', 119)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 32)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D32')
// CHECK:    ),
// CHECK:     # Symbol 33
// CHECK:    (('n_strx', 123)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 33)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D33')
// CHECK:    ),
// CHECK:     # Symbol 34
// CHECK:    (('n_strx', 127)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 34)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D34')
// CHECK:    ),
// CHECK:     # Symbol 35
// CHECK:    (('n_strx', 131)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 4)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D35')
// CHECK:    ),
// CHECK:     # Symbol 36
// CHECK:    (('n_strx', 135)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 4)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D36')
// CHECK:    ),
// CHECK:     # Symbol 37
// CHECK:    (('n_strx', 139)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 4)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D37')
// CHECK:    ),
// CHECK:     # Symbol 38
// CHECK:    (('n_strx', 143)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 35)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D38')
// CHECK:    ),
// CHECK:     # Symbol 39
// CHECK:    (('n_strx', 147)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 36)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'D39')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 40)
// CHECK:   ('iextdefsym', 40)
// CHECK:   ('nextdefsym', 0)
// CHECK:   ('iundefsym', 40)
// CHECK:   ('nundefsym', 0)
// CHECK:   ('tocoff', 0)
// CHECK:   ('ntoc', 0)
// CHECK:   ('modtaboff', 0)
// CHECK:   ('nmodtab', 0)
// CHECK:   ('extrefsymoff', 0)
// CHECK:   ('nextrefsyms', 0)
// CHECK:   ('indirectsymoff', 0)
// CHECK:   ('nindirectsyms', 0)
// CHECK:   ('extreloff', 0)
// CHECK:   ('nextrel', 0)
// CHECK:   ('locreloff', 0)
// CHECK:   ('nlocrel', 0)
// CHECK:   ('_indirect_symbols', [
// CHECK:   ])
// CHECK:  ),
// CHECK: ])
