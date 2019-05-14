// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --file-headers -S -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols | FileCheck -check-prefix CHECK-X86_32 %s
// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj --file-headers -S -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols | FileCheck -check-prefix CHECK-X86_64 %s

sym_local_B:
.globl sym_globl_def_B
.globl sym_globl_undef_B
sym_local_A:
.globl sym_globl_def_A
.globl sym_globl_undef_A
sym_local_C:
.globl sym_globl_def_C
.globl sym_globl_undef_C
        
sym_globl_def_A: 
sym_globl_def_B: 
sym_globl_def_C: 
Lsym_asm_temp:
        .long 0
        
// CHECK-X86_32: File: <stdin>
// CHECK-X86_32: Format: Mach-O 32-bit i386
// CHECK-X86_32: Arch: i386
// CHECK-X86_32: AddressSize: 32bit
// CHECK-X86_32: MachHeader {
// CHECK-X86_32:   Magic: Magic (0xFEEDFACE)
// CHECK-X86_32:   CpuType: X86 (0x7)
// CHECK-X86_32:   CpuSubType: CPU_SUBTYPE_I386_ALL (0x3)
// CHECK-X86_32:   FileType: Relocatable (0x1)
// CHECK-X86_32:   NumOfLoadCommands: 4
// CHECK-X86_32:   SizeOfLoadCommands: 244
// CHECK-X86_32:   Flags [ (0x0)
// CHECK-X86_32:   ]
// CHECK-X86_32: }
// CHECK-X86_32: Sections [
// CHECK-X86_32:   Section {
// CHECK-X86_32:     Index: 0
// CHECK-X86_32:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// CHECK-X86_32:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-X86_32:     Address: 0x0
// CHECK-X86_32:     Size: 0x4
// CHECK-X86_32:     Offset: 272
// CHECK-X86_32:     Alignment: 0
// CHECK-X86_32:     RelocationOffset: 0x0
// CHECK-X86_32:     RelocationCount: 0
// CHECK-X86_32:     Type: 0x0
// CHECK-X86_32:     Attributes [ (0x800000)
// CHECK-X86_32:       PureInstructions (0x800000)
// CHECK-X86_32:     ]
// CHECK-X86_32:     Reserved1: 0x0
// CHECK-X86_32:     Reserved2: 0x0
// CHECK-X86_32:   }
// CHECK-X86_32: ]
// CHECK-X86_32: Relocations [
// CHECK-X86_32: ]
// CHECK-X86_32: Symbols [
// CHECK-X86_32:   Symbol {
// CHECK-X86_32:     Name: sym_local_B (47)
// CHECK-X86_32:     Type: Section (0xE)
// CHECK-X86_32:     Section: __text (0x1)
// CHECK-X86_32:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_32:     Flags [ (0x0)
// CHECK-X86_32:     ]
// CHECK-X86_32:     Value: 0x0
// CHECK-X86_32:   }
// CHECK-X86_32:   Symbol {
// CHECK-X86_32:     Name: sym_local_A (93)
// CHECK-X86_32:     Type: Section (0xE)
// CHECK-X86_32:     Section: __text (0x1)
// CHECK-X86_32:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_32:     Flags [ (0x0)
// CHECK-X86_32:     ]
// CHECK-X86_32:     Value: 0x0
// CHECK-X86_32:   }
// CHECK-X86_32:   Symbol {
// CHECK-X86_32:     Name: sym_local_C (1)
// CHECK-X86_32:     Type: Section (0xE)
// CHECK-X86_32:     Section: __text (0x1)
// CHECK-X86_32:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_32:     Flags [ (0x0)
// CHECK-X86_32:     ]
// CHECK-X86_32:     Value: 0x0
// CHECK-X86_32:   }
// CHECK-X86_32:   Symbol {
// CHECK-X86_32:     Name: sym_globl_def_A (123)
// CHECK-X86_32:     Extern
// CHECK-X86_32:     Type: Section (0xE)
// CHECK-X86_32:     Section: __text (0x1)
// CHECK-X86_32:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_32:     Flags [ (0x0)
// CHECK-X86_32:     ]
// CHECK-X86_32:     Value: 0x0
// CHECK-X86_32:   }
// CHECK-X86_32:   Symbol {
// CHECK-X86_32:     Name: sym_globl_def_B (77)
// CHECK-X86_32:     Extern
// CHECK-X86_32:     Type: Section (0xE)
// CHECK-X86_32:     Section: __text (0x1)
// CHECK-X86_32:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_32:     Flags [ (0x0)
// CHECK-X86_32:     ]
// CHECK-X86_32:     Value: 0x0
// CHECK-X86_32:   }
// CHECK-X86_32:   Symbol {
// CHECK-X86_32:     Name: sym_globl_def_C (31)
// CHECK-X86_32:     Extern
// CHECK-X86_32:     Type: Section (0xE)
// CHECK-X86_32:     Section: __text (0x1)
// CHECK-X86_32:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_32:     Flags [ (0x0)
// CHECK-X86_32:     ]
// CHECK-X86_32:     Value: 0x0
// CHECK-X86_32:   }
// CHECK-X86_32:   Symbol {
// CHECK-X86_32:     Name: sym_globl_undef_A (105)
// CHECK-X86_32:     Extern
// CHECK-X86_32:     Type: Undef (0x0)
// CHECK-X86_32:     Section:  (0x0)
// CHECK-X86_32:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_32:     Flags [ (0x0)
// CHECK-X86_32:     ]
// CHECK-X86_32:     Value: 0x0
// CHECK-X86_32:   }
// CHECK-X86_32:   Symbol {
// CHECK-X86_32:     Name: sym_globl_undef_B (59)
// CHECK-X86_32:     Extern
// CHECK-X86_32:     Type: Undef (0x0)
// CHECK-X86_32:     Section:  (0x0)
// CHECK-X86_32:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_32:     Flags [ (0x0)
// CHECK-X86_32:     ]
// CHECK-X86_32:     Value: 0x0
// CHECK-X86_32:   }
// CHECK-X86_32:   Symbol {
// CHECK-X86_32:     Name: sym_globl_undef_C (13)
// CHECK-X86_32:     Extern
// CHECK-X86_32:     Type: Undef (0x0)
// CHECK-X86_32:     Section:  (0x0)
// CHECK-X86_32:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_32:     Flags [ (0x0)
// CHECK-X86_32:     ]
// CHECK-X86_32:     Value: 0x0
// CHECK-X86_32:   }
// CHECK-X86_32: ]
// CHECK-X86_32: Indirect Symbols {
// CHECK-X86_32:   Number: 0
// CHECK-X86_32:   Symbols [
// CHECK-X86_32:   ]
// CHECK-X86_32: }
// CHECK-X86_32: Segment {
// CHECK-X86_32:   Cmd: LC_SEGMENT
// CHECK-X86_32:   Name: 
// CHECK-X86_32:   Size: 124
// CHECK-X86_32:   vmaddr: 0x0
// CHECK-X86_32:   vmsize: 0x4
// CHECK-X86_32:   fileoff: 272
// CHECK-X86_32:   filesize: 4
// CHECK-X86_32:   maxprot: rwx
// CHECK-X86_32:   initprot: rwx
// CHECK-X86_32:   nsects: 1
// CHECK-X86_32:   flags: 0x0
// CHECK-X86_32: }
// CHECK-X86_32: Dysymtab {
// CHECK-X86_32:   ilocalsym: 0
// CHECK-X86_32:   nlocalsym: 3
// CHECK-X86_32:   iextdefsym: 3
// CHECK-X86_32:   nextdefsym: 3
// CHECK-X86_32:   iundefsym: 6
// CHECK-X86_32:   nundefsym: 3
// CHECK-X86_32:   tocoff: 0
// CHECK-X86_32:   ntoc: 0
// CHECK-X86_32:   modtaboff: 0
// CHECK-X86_32:   nmodtab: 0
// CHECK-X86_32:   extrefsymoff: 0
// CHECK-X86_32:   nextrefsyms: 0
// CHECK-X86_32:   indirectsymoff: 0
// CHECK-X86_32:   nindirectsyms: 0
// CHECK-X86_32:   extreloff: 0
// CHECK-X86_32:   nextrel: 0
// CHECK-X86_32:   locreloff: 0
// CHECK-X86_32:   nlocrel: 0
// CHECK-X86_32: }

// CHECK-X86_64: File: <stdin>
// CHECK-X86_64: Format: Mach-O 64-bit x86-64
// CHECK-X86_64: Arch: x86_64
// CHECK-X86_64: AddressSize: 64bit
// CHECK-X86_64: MachHeader {
// CHECK-X86_64:   Magic: Magic64 (0xFEEDFACF)
// CHECK-X86_64:   CpuType: X86-64 (0x1000007)
// CHECK-X86_64:   CpuSubType: CPU_SUBTYPE_X86_64_ALL (0x3)
// CHECK-X86_64:   FileType: Relocatable (0x1)
// CHECK-X86_64:   NumOfLoadCommands: 4
// CHECK-X86_64:   SizeOfLoadCommands: 272
// CHECK-X86_64:   Flags [ (0x0)
// CHECK-X86_64:   ]
// CHECK-X86_64:   Reserved: 0x0
// CHECK-X86_64: }
// CHECK-X86_64: Sections [
// CHECK-X86_64:   Section {
// CHECK-X86_64:     Index: 0
// CHECK-X86_64:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// CHECK-X86_64:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-X86_64:     Address: 0x0
// CHECK-X86_64:     Size: 0x4
// CHECK-X86_64:     Offset: 304
// CHECK-X86_64:     Alignment: 0
// CHECK-X86_64:     RelocationOffset: 0x0
// CHECK-X86_64:     RelocationCount: 0
// CHECK-X86_64:     Type: 0x0
// CHECK-X86_64:     Attributes [ (0x800000)
// CHECK-X86_64:       PureInstructions (0x800000)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Reserved1: 0x0
// CHECK-X86_64:     Reserved2: 0x0
// CHECK-X86_64:     Reserved3: 0x0
// CHECK-X86_64:   }
// CHECK-X86_64: ]
// CHECK-X86_64: Relocations [
// CHECK-X86_64: ]
// CHECK-X86_64: Symbols [
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: sym_local_B (47)
// CHECK-X86_64:     Type: Section (0xE)
// CHECK-X86_64:     Section: __text (0x1)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x0
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: sym_local_A (93)
// CHECK-X86_64:     Type: Section (0xE)
// CHECK-X86_64:     Section: __text (0x1)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x0
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: sym_local_C (1)
// CHECK-X86_64:     Type: Section (0xE)
// CHECK-X86_64:     Section: __text (0x1)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x0
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: sym_globl_def_A (123)
// CHECK-X86_64:     Extern
// CHECK-X86_64:     Type: Section (0xE)
// CHECK-X86_64:     Section: __text (0x1)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x0
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: sym_globl_def_B (77)
// CHECK-X86_64:     Extern
// CHECK-X86_64:     Type: Section (0xE)
// CHECK-X86_64:     Section: __text (0x1)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x0
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: sym_globl_def_C (31)
// CHECK-X86_64:     Extern
// CHECK-X86_64:     Type: Section (0xE)
// CHECK-X86_64:     Section: __text (0x1)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x0
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: sym_globl_undef_A (105)
// CHECK-X86_64:     Extern
// CHECK-X86_64:     Type: Undef (0x0)
// CHECK-X86_64:     Section:  (0x0)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x0
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: sym_globl_undef_B (59)
// CHECK-X86_64:     Extern
// CHECK-X86_64:     Type: Undef (0x0)
// CHECK-X86_64:     Section:  (0x0)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x0
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: sym_globl_undef_C (13)
// CHECK-X86_64:     Extern
// CHECK-X86_64:     Type: Undef (0x0)
// CHECK-X86_64:     Section:  (0x0)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x0
// CHECK-X86_64:   }
// CHECK-X86_64: ]
// CHECK-X86_64: Indirect Symbols {
// CHECK-X86_64:   Number: 0
// CHECK-X86_64:   Symbols [
// CHECK-X86_64:   ]
// CHECK-X86_64: }
// CHECK-X86_64: Segment {
// CHECK-X86_64:   Cmd: LC_SEGMENT_64
// CHECK-X86_64:   Name: 
// CHECK-X86_64:   Size: 152
// CHECK-X86_64:   vmaddr: 0x0
// CHECK-X86_64:   vmsize: 0x4
// CHECK-X86_64:   fileoff: 304
// CHECK-X86_64:   filesize: 4
// CHECK-X86_64:   maxprot: rwx
// CHECK-X86_64:   initprot: rwx
// CHECK-X86_64:   nsects: 1
// CHECK-X86_64:   flags: 0x0
// CHECK-X86_64: }
// CHECK-X86_64: Dysymtab {
// CHECK-X86_64:   ilocalsym: 0
// CHECK-X86_64:   nlocalsym: 3
// CHECK-X86_64:   iextdefsym: 3
// CHECK-X86_64:   nextdefsym: 3
// CHECK-X86_64:   iundefsym: 6
// CHECK-X86_64:   nundefsym: 3
// CHECK-X86_64:   tocoff: 0
// CHECK-X86_64:   ntoc: 0
// CHECK-X86_64:   modtaboff: 0
// CHECK-X86_64:   nmodtab: 0
// CHECK-X86_64:   extrefsymoff: 0
// CHECK-X86_64:   nextrefsyms: 0
// CHECK-X86_64:   indirectsymoff: 0
// CHECK-X86_64:   nindirectsyms: 0
// CHECK-X86_64:   extreloff: 0
// CHECK-X86_64:   nextrel: 0
// CHECK-X86_64:   locreloff: 0
// CHECK-X86_64:   nlocrel: 0
// CHECK-X86_64: }
