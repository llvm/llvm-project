// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --file-headers -S --sd -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols - | FileCheck %s

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

// CHECK: File: <stdin>
// CHECK: Format: Mach-O 32-bit i386
// CHECK: Arch: i386
// CHECK: AddressSize: 32bit
// CHECK: MachHeader {
// CHECK:   Magic: Magic (0xFEEDFACE)
// CHECK:   CpuType: X86 (0x7)
// CHECK:   CpuSubType: CPU_SUBTYPE_I386_ALL (0x3)
// CHECK:   FileType: Relocatable (0x1)
// CHECK:   NumOfLoadCommands: 4
// CHECK:   SizeOfLoadCommands: 2624
// CHECK:   Flags [ (0x0)
// CHECK:   ]
// CHECK: }
// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Index: 0
// CHECK:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x800000)
// CHECK:       PureInstructions (0x800000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 1
// CHECK:     Name: __const (5F 5F 63 6F 6E 73 74 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 2
// CHECK:     Name: __static_const (5F 5F 73 74 61 74 69 63 5F 63 6F 6E 73 74 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 3
// CHECK:     Name: __cstring (5F 5F 63 73 74 72 69 6E 67 00 00 00 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: CStringLiterals (0x2)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 4
// CHECK:     Name: __literal4 (5F 5F 6C 69 74 65 72 61 6C 34 00 00 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 2
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: 4ByteLiterals (0x3)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 5
// CHECK:     Name: __literal8 (5F 5F 6C 69 74 65 72 61 6C 38 00 00 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 3
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: 8ByteLiterals (0x4)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 6
// CHECK:     Name: __literal16 (5F 5F 6C 69 74 65 72 61 6C 31 36 00 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 4
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: 16ByteLiterals (0xE)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 7
// CHECK:     Name: __constructor (5F 5F 63 6F 6E 73 74 72 75 63 74 6F 72 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 8
// CHECK:     Name: __destructor (5F 5F 64 65 73 74 72 75 63 74 6F 72 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 9
// CHECK:     Name: __symbol_stub (5F 5F 73 79 6D 62 6F 6C 5F 73 74 75 62 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: SymbolStubs (0x8)
// CHECK:     Attributes [ (0x800000)
// CHECK:       PureInstructions (0x800000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x10
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 10
// CHECK:     Name: __picsymbol_stub (5F 5F 70 69 63 73 79 6D 62 6F 6C 5F 73 74 75 62)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: SymbolStubs (0x8)
// CHECK:     Attributes [ (0x800000)
// CHECK:       PureInstructions (0x800000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x1A
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 11
// CHECK:     Name: __data (5F 5F 64 61 74 61 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 12
// CHECK:     Name: __static_data (5F 5F 73 74 61 74 69 63 5F 64 61 74 61 00 00 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 13
// CHECK:     Name: __nl_symbol_ptr (5F 5F 6E 6C 5F 73 79 6D 62 6F 6C 5F 70 74 72 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 2
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: NonLazySymbolPointers (0x6)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 14
// CHECK:     Name: __la_symbol_ptr (5F 5F 6C 61 5F 73 79 6D 62 6F 6C 5F 70 74 72 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 2
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: LazySymbolPointers (0x7)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 15
// CHECK:     Name: __dyld (5F 5F 64 79 6C 64 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 16
// CHECK:     Name: __mod_init_func (5F 5F 6D 6F 64 5F 69 6E 69 74 5F 66 75 6E 63 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 2
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: ModInitFuncPointers (0x9)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 17
// CHECK:     Name: __mod_term_func (5F 5F 6D 6F 64 5F 74 65 72 6D 5F 66 75 6E 63 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 2
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: ModTermFuncPointers (0xA)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 18
// CHECK:     Name: __const (5F 5F 63 6F 6E 73 74 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 19
// CHECK:     Name: __class (5F 5F 63 6C 61 73 73 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 20
// CHECK:     Name: __meta_class (5F 5F 6D 65 74 61 5F 63 6C 61 73 73 00 00 00 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 21
// CHECK:     Name: __cat_cls_meth (5F 5F 63 61 74 5F 63 6C 73 5F 6D 65 74 68 00 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 22
// CHECK:     Name: __cat_inst_meth (5F 5F 63 61 74 5F 69 6E 73 74 5F 6D 65 74 68 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 23
// CHECK:     Name: __protocol (5F 5F 70 72 6F 74 6F 63 6F 6C 00 00 00 00 00 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 24
// CHECK:     Name: __string_object (5F 5F 73 74 72 69 6E 67 5F 6F 62 6A 65 63 74 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 25
// CHECK:     Name: __cls_meth (5F 5F 63 6C 73 5F 6D 65 74 68 00 00 00 00 00 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 26
// CHECK:     Name: __inst_meth (5F 5F 69 6E 73 74 5F 6D 65 74 68 00 00 00 00 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 27
// CHECK:     Name: __cls_refs (5F 5F 63 6C 73 5F 72 65 66 73 00 00 00 00 00 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 2
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: LiteralPointers (0x5)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 28
// CHECK:     Name: __message_refs (5F 5F 6D 65 73 73 61 67 65 5F 72 65 66 73 00 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 2
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: LiteralPointers (0x5)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 29
// CHECK:     Name: __symbols (5F 5F 73 79 6D 62 6F 6C 73 00 00 00 00 00 00 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 30
// CHECK:     Name: __category (5F 5F 63 61 74 65 67 6F 72 79 00 00 00 00 00 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 31
// CHECK:     Name: __class_vars (5F 5F 63 6C 61 73 73 5F 76 61 72 73 00 00 00 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 32
// CHECK:     Name: __instance_vars (5F 5F 69 6E 73 74 61 6E 63 65 5F 76 61 72 73 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 33
// CHECK:     Name: __module_info (5F 5F 6D 6F 64 75 6C 65 5F 69 6E 66 6F 00 00 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x100000)
// CHECK:       NoDeadStrip (0x100000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 34
// CHECK:     Name: __selector_strs (5F 5F 73 65 6C 65 63 74 6F 72 5F 73 74 72 73 00)
// CHECK:     Segment: __OBJC (5F 5F 4F 42 4A 43 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: CStringLiterals (0x2)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 35
// CHECK:     Name: __picsymbolstub4 (5F 5F 70 69 63 73 79 6D 62 6F 6C 73 74 75 62 34)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 2652
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: SymbolStubs (0x8)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x10
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: D0 (136)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D1 (121)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D2 (106)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __const (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D3 (91)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __static_const (0x3)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D4 (76)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __cstring (0x4)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D5 (61)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __literal4 (0x5)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D6 (46)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __literal8 (0x6)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D7 (31)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __literal16 (0x7)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D8 (16)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __constructor (0x8)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D9 (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __destructor (0x9)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D10 (147)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __symbol_stub (0xA)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D11 (132)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __picsymbol_stub (0xB)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D12 (117)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0xC)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D13 (102)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __static_data (0xD)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D14 (87)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __nl_symbol_ptr (0xE)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D15 (72)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __la_symbol_ptr (0xF)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D16 (57)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __dyld (0x10)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D17 (42)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __mod_init_func (0x11)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D18 (27)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __mod_term_func (0x12)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D19 (12)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __const (0x13)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D20 (143)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __class (0x14)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D21 (128)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __meta_class (0x15)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D22 (113)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __cat_cls_meth (0x16)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D23 (98)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __cat_inst_meth (0x17)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D24 (83)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __protocol (0x18)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D25 (68)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __string_object (0x19)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D26 (53)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __cls_meth (0x1A)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D27 (38)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __inst_meth (0x1B)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D28 (23)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __cls_refs (0x1C)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D29 (8)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __message_refs (0x1D)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D30 (139)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __symbols (0x1E)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D31 (124)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __category (0x1F)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D32 (109)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __class_vars (0x20)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D33 (94)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __instance_vars (0x21)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D34 (79)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __module_info (0x22)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D35 (64)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __cstring (0x4)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D36 (49)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __cstring (0x4)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D37 (34)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __cstring (0x4)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D38 (19)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __selector_strs (0x23)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: D39 (4)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __picsymbolstub4 (0x24)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK: ]
// CHECK: Indirect Symbols {
// CHECK:   Number: 0
// CHECK:   Symbols [
// CHECK:   ]
// CHECK: }
// CHECK: Segment {
// CHECK:   Cmd: LC_SEGMENT
// CHECK:   Name: 
// CHECK:   Size: 2504
// CHECK:   vmaddr: 0x0
// CHECK:   vmsize: 0x0
// CHECK:   fileoff: 2652
// CHECK:   filesize: 0
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 36
// CHECK:   flags: 0x0
// CHECK: }
// CHECK: Dysymtab {
// CHECK:   ilocalsym: 0
// CHECK:   nlocalsym: 40
// CHECK:   iextdefsym: 40
// CHECK:   nextdefsym: 0
// CHECK:   iundefsym: 40
// CHECK:   nundefsym: 0
// CHECK:   tocoff: 0
// CHECK:   ntoc: 0
// CHECK:   modtaboff: 0
// CHECK:   nmodtab: 0
// CHECK:   extrefsymoff: 0
// CHECK:   nextrefsyms: 0
// CHECK:   indirectsymoff: 0
// CHECK:   nindirectsyms: 0
// CHECK:   extreloff: 0
// CHECK:   nextrel: 0
// CHECK:   locreloff: 0
// CHECK:   nlocrel: 0
// CHECK: }
