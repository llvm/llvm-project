// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --file-headers -S -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols - | FileCheck %s

        .reference sym_ref_A
        .reference sym_ref_def_A
sym_ref_def_A:
sym_ref_def_C:  
        .reference sym_ref_def_C
        .reference sym_ref_def_D
        .globl sym_ref_def_D
        .globl sym_ref_def_E
        .reference sym_ref_def_E
        
        .weak_reference sym_weak_ref_A
        .weak_reference sym_weak_ref_def_A
sym_weak_ref_def_A:        
sym_weak_ref_def_B:
        .weak_reference sym_weak_ref_def_B

        .data
        .globl sym_weak_def_A
        .weak_definition sym_weak_def_A        
sym_weak_def_A:
sym_weak_def_B:
        .weak_definition sym_weak_def_B
        .globl sym_weak_def_B
        .weak_definition sym_weak_def_C
sym_weak_def_C:
        .globl sym_weak_def_C

        .lazy_reference sym_lazy_ref_A
        .lazy_reference sym_lazy_ref_B
sym_lazy_ref_B:
sym_lazy_ref_C:
        .lazy_reference sym_lazy_ref_C
        .lazy_reference sym_lazy_ref_D
        .globl sym_lazy_ref_D
        .globl sym_lazy_ref_E
        .lazy_reference sym_lazy_ref_E

        .private_extern sym_private_ext_A
        .private_extern sym_private_ext_B
sym_private_ext_B:
sym_private_ext_C:
        .private_extern sym_private_ext_C
        .private_extern sym_private_ext_D
        .globl sym_private_ext_D
        .globl sym_private_ext_E
        .private_extern sym_private_ext_E

        .no_dead_strip sym_no_dead_strip_A

sym_symbol_resolver_A:
	.symbol_resolver sym_symbol_resolver_A

        .reference sym_ref_A
        .desc sym_ref_A, 1
        .desc sym_ref_A, 0x1234

        .desc sym_desc_flags,0x47
sym_desc_flags:
        
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
// CHECK:   SizeOfLoadCommands: 312
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
// CHECK:     Offset: 340
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x800000)
// CHECK:       PureInstructions (0x800000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 1
// CHECK:     Name: __data (5F 5F 64 61 74 61 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x0
// CHECK:     Offset: 340
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: sym_ref_def_A (354)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x20)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_ref_def_C (158)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x20)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_weak_ref_def_A (368)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x40)
// CHECK:       WeakRef (0x40)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_weak_ref_def_B (220)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_lazy_ref_B (190)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x20)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_lazy_ref_C (128)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x20)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_symbol_resolver_A (257)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x100)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_desc_flags (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x40)
// CHECK:       WeakRef (0x40)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_private_ext_B (172)
// CHECK:     PrivateExtern
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_private_ext_C (110)
// CHECK:     PrivateExtern
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_weak_def_A (339)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x80)
// CHECK:       WeakDef (0x80)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_weak_def_B (205)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x80)
// CHECK:       WeakDef (0x80)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_weak_def_C (143)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x80)
// CHECK:       WeakDef (0x80)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_lazy_ref_A (299)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: ReferenceFlagUndefinedLazy (0x1)
// CHECK:     Flags [ (0x20)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_lazy_ref_D (81)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x20)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_lazy_ref_E (34)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: ReferenceFlagUndefinedLazy (0x1)
// CHECK:     Flags [ (0x20)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_no_dead_strip_A (279)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x20)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_private_ext_A (239)
// CHECK:     PrivateExtern
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_private_ext_D (63)
// CHECK:     PrivateExtern
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_private_ext_E (16)
// CHECK:     PrivateExtern
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_ref_A (314)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: ReferenceFlagPrivateUndefinedNonLazy (0x4)
// CHECK:     Flags [ (0x1230)
// CHECK:       NoDeadStrip (0x20)
// CHECK:       ReferencedDynamically (0x10)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_ref_def_D (96)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x20)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_ref_def_E (49)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x20)
// CHECK:       NoDeadStrip (0x20)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_weak_ref_A (324)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x40)
// CHECK:       WeakRef (0x40)
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
// CHECK:   Size: 192
// CHECK:   vmaddr: 0x0
// CHECK:   vmsize: 0x0
// CHECK:   fileoff: 340
// CHECK:   filesize: 0
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 2
// CHECK:   flags: 0x0
// CHECK: }
// CHECK: Dysymtab {
// CHECK:   ilocalsym: 0
// CHECK:   nlocalsym: 8
// CHECK:   iextdefsym: 8
// CHECK:   nextdefsym: 5
// CHECK:   iundefsym: 13
// CHECK:   nundefsym: 11
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
