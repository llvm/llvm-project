// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --file-headers -S -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols - | FileCheck %s

// FIXME: We are missing a lot of diagnostics on this kind of stuff which the
// assembler has.
        
        .lazy_symbol_pointer
        .indirect_symbol sym_lsp_B
        .long 0
        
        .globl sym_lsp_A
        .indirect_symbol sym_lsp_A
        .long 0
        
sym_lsp_C:      
        .indirect_symbol sym_lsp_C
        .long 0

// FIXME: Enable this test once missing llvm-mc support is in place.
.if 0
        .indirect_symbol sym_lsp_D
        .long sym_lsp_D
.endif

        .indirect_symbol sym_lsp_E
        .long 0xFA

// FIXME: Enable this test once missing llvm-mc support is in place.
.if 0
sym_lsp_F = 10
        .indirect_symbol sym_lsp_F
        .long 0
.endif

        .globl sym_lsp_G
sym_lsp_G:
        .indirect_symbol sym_lsp_G
        .long 0
        
        .non_lazy_symbol_pointer
        .indirect_symbol sym_nlp_B
        .long 0

        .globl sym_nlp_A
        .indirect_symbol sym_nlp_A
        .long 0

sym_nlp_C:      
        .indirect_symbol sym_nlp_C
        .long 0

// FIXME: Enable this test once missing llvm-mc support is in place.
.if 0
        .indirect_symbol sym_nlp_D
        .long sym_nlp_D
.endif

        .indirect_symbol sym_nlp_E
        .long 0xAF

// FIXME: Enable this test once missing llvm-mc support is in place.
.if 0
sym_nlp_F = 10
        .indirect_symbol sym_nlp_F
        .long 0
.endif

        .globl sym_nlp_G
sym_nlp_G:
        .indirect_symbol sym_nlp_G
        .long 0

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
// CHECK:   SizeOfLoadCommands: 380
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
// CHECK:     Offset: 408
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
// CHECK:     Name: __la_symbol_ptr (5F 5F 6C 61 5F 73 79 6D 62 6F 6C 5F 70 74 72 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x14
// CHECK:     Offset: 408
// CHECK:     Alignment: 2
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: LazySymbolPointers (0x7)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 2
// CHECK:     Name: __nl_symbol_ptr (5F 5F 6E 6C 5F 73 79 6D 62 6F 6C 5F 70 74 72 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x14
// CHECK:     Size: 0x14
// CHECK:     Offset: 428
// CHECK:     Alignment: 2
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: NonLazySymbolPointers (0x6)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x5
// CHECK:     Reserved2: 0x0
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: sym_lsp_C (41)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __la_symbol_ptr (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x8
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_nlp_C (51)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __nl_symbol_ptr (0x3)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x1C
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_lsp_G (1)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __la_symbol_ptr (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x10
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_nlp_G (11)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __nl_symbol_ptr (0x3)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x24
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_lsp_A (81)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_lsp_B (61)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: ReferenceFlagUndefinedLazy (0x1)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_lsp_E (21)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: ReferenceFlagUndefinedLazy (0x1)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_nlp_A (91)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_nlp_B (71)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: sym_nlp_E (31)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK: ]
// CHECK: Indirect Symbols {
// CHECK:   Number: 10
// CHECK:   Symbols [
// CHECK:     Entry {
// CHECK:       Entry Index: 0
// CHECK:       Symbol Index: 0x5
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 1
// CHECK:       Symbol Index: 0x4
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 2
// CHECK:       Symbol Index: 0x0
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 3
// CHECK:       Symbol Index: 0x6
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 4
// CHECK:       Symbol Index: 0x2
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 5
// CHECK:       Symbol Index: 0x8
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 6
// CHECK:       Symbol Index: 0x7
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 7
// CHECK:       Symbol Index: 0x80000000
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 8
// CHECK:       Symbol Index: 0x9
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 9
// CHECK:       Symbol Index: 0x3
// CHECK:     }
// CHECK:   ]
// CHECK: }
// CHECK: Segment {
// CHECK:   Cmd: LC_SEGMENT
// CHECK:   Name: 
// CHECK:   Size: 260
// CHECK:   vmaddr: 0x0
// CHECK:   vmsize: 0x28
// CHECK:   fileoff: 408
// CHECK:   filesize: 40
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 3
// CHECK:   flags: 0x0
// CHECK: }
// CHECK: Dysymtab {
// CHECK:   ilocalsym: 0
// CHECK:   nlocalsym: 2
// CHECK:   iextdefsym: 2
// CHECK:   nextdefsym: 2
// CHECK:   iundefsym: 4
// CHECK:   nundefsym: 6
// CHECK:   tocoff: 0
// CHECK:   ntoc: 0
// CHECK:   modtaboff: 0
// CHECK:   nmodtab: 0
// CHECK:   extrefsymoff: 0
// CHECK:   nextrefsyms: 0
// CHECK:   indirectsymoff: 448
// CHECK:   nindirectsyms: 10
// CHECK:   extreloff: 0
// CHECK:   nextrel: 0
// CHECK:   locreloff: 0
// CHECK:   nlocrel: 0
// CHECK: }
