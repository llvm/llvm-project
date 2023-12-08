// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --file-headers -S --sd -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols - | FileCheck %s

_b:
        _c = 0
_e:
        _f = 0
        
	.section	__IMPORT,__jump_table,symbol_stubs,pure_instructions+self_modifying_code,5
.indirect_symbol _a
	.ascii	 "\364\364\364\364\364"        
.indirect_symbol _b
	.ascii	 "\364\364\364\364\364"        
.indirect_symbol _c
	.ascii	 "\364\364\364\364\364"        
	.section	__IMPORT,__pointers,non_lazy_symbol_pointers
.indirect_symbol _d
	.long	0
.indirect_symbol _e
	.long	0
.indirect_symbol _f
	.long	0

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
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 1
// CHECK:     Name: __jump_table (5F 5F 6A 75 6D 70 5F 74 61 62 6C 65 00 00 00 00)
// CHECK:     Segment: __IMPORT (5F 5F 49 4D 50 4F 52 54 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0xF
// CHECK:     Offset: 408
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: SymbolStubs (0x8)
// CHECK:     Attributes [ (0x840000)
// CHECK:       PureInstructions (0x800000)
// CHECK:       SelfModifyingCode (0x40000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x5
// CHECK:     SectionData (
// CHECK:       0000: F4F4F4F4 F4F4F4F4 F4F4F4F4 F4F4F4    |...............|
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 2
// CHECK:     Name: __pointers (5F 5F 70 6F 69 6E 74 65 72 73 00 00 00 00 00 00)
// CHECK:     Segment: __IMPORT (5F 5F 49 4D 50 4F 52 54 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0xF
// CHECK:     Size: 0xC
// CHECK:     Offset: 423
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: NonLazySymbolPointers (0x6)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x3
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 00000000 00000000 00000000           |............|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: _b (13)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _c (10)
// CHECK:     Type: Abs (0x2)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _e (4)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _f (1)
// CHECK:     Type: Abs (0x2)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _a (16)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
// CHECK:     RefType: ReferenceFlagUndefinedLazy (0x1)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _d (7)
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
// CHECK:   Number: 6
// CHECK:   Symbols [
// CHECK:     Entry {
// CHECK:       Entry Index: 0
// CHECK:       Symbol Index: 0x4
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 1
// CHECK:       Symbol Index: 0x0
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 2
// CHECK:       Symbol Index: 0x1
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 3
// CHECK:       Symbol Index: 0x5
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 4
// CHECK:       Symbol Index: 0x80000000
// CHECK:     }
// CHECK:     Entry {
// CHECK:       Entry Index: 5
// CHECK:       Symbol Index: 0xC0000000
// CHECK:     }
// CHECK:   ]
// CHECK: }
// CHECK: Segment {
// CHECK:   Cmd: LC_SEGMENT
// CHECK:   Name: 
// CHECK:   Size: 260
// CHECK:   vmaddr: 0x0
// CHECK:   vmsize: 0x1B
// CHECK:   fileoff: 408
// CHECK:   filesize: 27
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 3
// CHECK:   flags: 0x0
// CHECK: }
// CHECK: Dysymtab {
// CHECK:   ilocalsym: 0
// CHECK:   nlocalsym: 4
// CHECK:   iextdefsym: 4
// CHECK:   nextdefsym: 0
// CHECK:   iundefsym: 4
// CHECK:   nundefsym: 2
// CHECK:   tocoff: 0
// CHECK:   ntoc: 0
// CHECK:   modtaboff: 0
// CHECK:   nmodtab: 0
// CHECK:   extrefsymoff: 0
// CHECK:   nextrefsyms: 0
// CHECK:   indirectsymoff: 436
// CHECK:   nindirectsyms: 6
// CHECK:   extreloff: 0
// CHECK:   nextrel: 0
// CHECK:   locreloff: 0
// CHECK:   nlocrel: 0
// CHECK: }
