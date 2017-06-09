// RUN: llvm-mc -triple i386-apple-darwin10 %s -filetype=obj -o %t.o
// RUN: llvm-readobj -file-headers -s -sd -r -t -macho-segment -macho-dysymtab -macho-indirect-symbols < %t.o > %t.dump
// RUN: FileCheck --check-prefix=CHECK-I386 < %t.dump %s

// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o %t.o
// RUN: llvm-readobj -file-headers -s -sd -r -t -macho-segment -macho-dysymtab -macho-indirect-symbols < %t.o > %t.dump
// RUN: FileCheck --check-prefix=CHECK-X86_64 < %t.dump %s

.data

        .long 0
a:
        .long 0
b = a

c:      .long b

d2 = d
.globl d2
d3 = d + 4
.globl d3

e = a + 4

g:
f = g
        .long 0
        
        .long b
        .long e
        .long a + 4
        .long d
        .long d2
        .long d3
        .long f
        .long g

///
        .text
t0:
Lt0_a:
        ret

	.data
Lt0_b:
Lt0_x = Lt0_a - Lt0_b
	.quad	Lt0_x

// CHECK-I386: File: <stdin>
// CHECK-I386: Format: Mach-O 32-bit i386
// CHECK-I386: Arch: i386
// CHECK-I386: AddressSize: 32bit
// CHECK-I386: MachHeader {
// CHECK-I386:   Magic: Magic (0xFEEDFACE)
// CHECK-I386:   CpuType: X86 (0x7)
// CHECK-I386:   CpuSubType: CPU_SUBTYPE_I386_ALL (0x3)
// CHECK-I386:   FileType: Relocatable (0x1)
// CHECK-I386:   NumOfLoadCommands: 4
// CHECK-I386:   SizeOfLoadCommands: 312
// CHECK-I386:   Flags [ (0x0)
// CHECK-I386:   ]
// CHECK-I386: }
// CHECK-I386: Sections [
// CHECK-I386:   Section {
// CHECK-I386:     Index: 0
// CHECK-I386:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// CHECK-I386:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-I386:     Address: 0x0
// CHECK-I386:     Size: 0x1
// CHECK-I386:     Offset: 340
// CHECK-I386:     Alignment: 0
// CHECK-I386:     RelocationOffset: 0x0
// CHECK-I386:     RelocationCount: 0
// CHECK-I386:     Type: 0x0
// CHECK-I386:     Attributes [ (0x800004)
// CHECK-I386:       PureInstructions (0x800000)
// CHECK-I386:       SomeInstructions (0x4)
// CHECK-I386:     ]
// CHECK-I386:     Reserved1: 0x0
// CHECK-I386:     Reserved2: 0x0
// CHECK-I386:     SectionData (
// CHECK-I386:       0000: C3                                   |.|
// CHECK-I386:     )
// CHECK-I386:   }
// CHECK-I386:   Section {
// CHECK-I386:     Index: 1
// CHECK-I386:     Name: __data (5F 5F 64 61 74 61 00 00 00 00 00 00 00 00 00 00)
// CHECK-I386:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK-I386:     Address: 0x1
// CHECK-I386:     Size: 0x38
// CHECK-I386:     Offset: 341
// CHECK-I386:     Alignment: 0
// CHECK-I386:     RelocationOffset: 0x190
// CHECK-I386:     RelocationCount: 9
// CHECK-I386:     Type: 0x0
// CHECK-I386:     Attributes [ (0x0)
// CHECK-I386:     ]
// CHECK-I386:     Reserved1: 0x0
// CHECK-I386:     Reserved2: 0x0
// CHECK-I386:     SectionData (
// CHECK-I386:       0000: 00000000 00000000 05000000 00000000  |................|
// CHECK-I386:       0010: 05000000 09000000 09000000 00000000  |................|
// CHECK-I386:       0020: 00000000 00000000 0D000000 0D000000  |................|
// CHECK-I386:       0030: CFFFFFFF FFFFFFFF                    |........|
// CHECK-I386:     )
// CHECK-I386:   }
// CHECK-I386: ]
// CHECK-I386: Relocations [
// CHECK-I386:   Section __data {
// CHECK-I386:     0x2C 0 2 0 GENERIC_RELOC_VANILLA 0 __data
// CHECK-I386:     0x28 0 2 0 GENERIC_RELOC_VANILLA 0 __data
// CHECK-I386:     0x24 0 2 1 GENERIC_RELOC_VANILLA 0 d3
// CHECK-I386:     0x20 0 2 1 GENERIC_RELOC_VANILLA 0 d{{$}}
// CHECK-I386:     0x1C 0 2 1 GENERIC_RELOC_VANILLA 0 d{{$}}
// CHECK-I386:     0x18 0 2 n/a GENERIC_RELOC_VANILLA 1 0x5
// CHECK-I386:     0x14 0 2 0 GENERIC_RELOC_VANILLA 0 __data
// CHECK-I386:     0x10 0 2 0 GENERIC_RELOC_VANILLA 0 __data
// CHECK-I386:     0x8 0 2 0 GENERIC_RELOC_VANILLA 0 __data
// CHECK-I386:   }
// CHECK-I386: ]
// CHECK-I386: Symbols [
// CHECK-I386:   Symbol {
// CHECK-I386:     Name: a (13)
// CHECK-I386:     Type: Section (0xE)
// CHECK-I386:     Section: __data (0x2)
// CHECK-I386:     RefType: UndefinedNonLazy (0x0)
// CHECK-I386:     Flags [ (0x0)
// CHECK-I386:     ]
// CHECK-I386:     Value: 0x5
// CHECK-I386:   }
// CHECK-I386:   Symbol {
// CHECK-I386:     Name: b (11)
// CHECK-I386:     Type: Section (0xE)
// CHECK-I386:     Section: __data (0x2)
// CHECK-I386:     RefType: UndefinedNonLazy (0x0)
// CHECK-I386:     Flags [ (0x0)
// CHECK-I386:     ]
// CHECK-I386:     Value: 0x5
// CHECK-I386:   }
// CHECK-I386:   Symbol {
// CHECK-I386:     Name: c (9)
// CHECK-I386:     Type: Section (0xE)
// CHECK-I386:     Section: __data (0x2)
// CHECK-I386:     RefType: UndefinedNonLazy (0x0)
// CHECK-I386:     Flags [ (0x0)
// CHECK-I386:     ]
// CHECK-I386:     Value: 0x9
// CHECK-I386:   }
// CHECK-I386:   Symbol {
// CHECK-I386:     Name: e (5)
// CHECK-I386:     Type: Section (0xE)
// CHECK-I386:     Section: __data (0x2)
// CHECK-I386:     RefType: UndefinedNonLazy (0x0)
// CHECK-I386:     Flags [ (0x200)
// CHECK-I386:     ]
// CHECK-I386:     Value: 0x9
// CHECK-I386:   }
// CHECK-I386:   Symbol {
// CHECK-I386:     Name: g (1)
// CHECK-I386:     Type: Section (0xE)
// CHECK-I386:     Section: __data (0x2)
// CHECK-I386:     RefType: UndefinedNonLazy (0x0)
// CHECK-I386:     Flags [ (0x0)
// CHECK-I386:     ]
// CHECK-I386:     Value: 0xD
// CHECK-I386:   }
// CHECK-I386:   Symbol {
// CHECK-I386:     Name: f (3)
// CHECK-I386:     Type: Section (0xE)
// CHECK-I386:     Section: __data (0x2)
// CHECK-I386:     RefType: UndefinedNonLazy (0x0)
// CHECK-I386:     Flags [ (0x0)
// CHECK-I386:     ]
// CHECK-I386:     Value: 0xD
// CHECK-I386:   }
// CHECK-I386:   Symbol {
// CHECK-I386:     Name: t0 (21)
// CHECK-I386:     Type: Section (0xE)
// CHECK-I386:     Section: __text (0x1)
// CHECK-I386:     RefType: UndefinedNonLazy (0x0)
// CHECK-I386:     Flags [ (0x0)
// CHECK-I386:     ]
// CHECK-I386:     Value: 0x0
// CHECK-I386:   }
// CHECK-I386:   Symbol {
// CHECK-I386:     Name: d (7)
// CHECK-I386:     Extern
// CHECK-I386:     Type: Undef (0x0)
// CHECK-I386:     Section:  (0x0)
// CHECK-I386:     RefType: UndefinedNonLazy (0x0)
// CHECK-I386:     Flags [ (0x0)
// CHECK-I386:     ]
// CHECK-I386:     Value: 0x0
// CHECK-I386:   }
// CHECK-I386:   Symbol {
// CHECK-I386:     Name: d2 (18)
// CHECK-I386:     Extern
// CHECK-I386:     Type: Indirect (0xA)
// CHECK-I386:     Section:  (0x0)
// CHECK-I386:     RefType: UndefinedNonLazy (0x0)
// CHECK-I386:     Flags [ (0x0)
// CHECK-I386:     ]
// CHECK-I386:     Value: 0x7
// CHECK-I386:   }
// CHECK-I386:   Symbol {
// CHECK-I386:     Name: d3 (15)
// CHECK-I386:     Extern
// CHECK-I386:     Type: Undef (0x0)
// CHECK-I386:     Section:  (0x0)
// CHECK-I386:     RefType: UndefinedNonLazy (0x0)
// CHECK-I386:     Flags [ (0x200)
// CHECK-I386:     ]
// CHECK-I386:     Value: 0x0
// CHECK-I386:   }
// CHECK-I386: ]
// CHECK-I386: Indirect Symbols {
// CHECK-I386:   Number: 0
// CHECK-I386:   Symbols [
// CHECK-I386:   ]
// CHECK-I386: }
// CHECK-I386: Segment {
// CHECK-I386:   Cmd: LC_SEGMENT
// CHECK-I386:   Name: 
// CHECK-I386:   Size: 192
// CHECK-I386:   vmaddr: 0x0
// CHECK-I386:   vmsize: 0x39
// CHECK-I386:   fileoff: 340
// CHECK-I386:   filesize: 57
// CHECK-I386:   maxprot: rwx
// CHECK-I386:   initprot: rwx
// CHECK-I386:   nsects: 2
// CHECK-I386:   flags: 0x0
// CHECK-I386: }
// CHECK-I386: Dysymtab {
// CHECK-I386:   ilocalsym: 0
// CHECK-I386:   nlocalsym: 7
// CHECK-I386:   iextdefsym: 7
// CHECK-I386:   nextdefsym: 0
// CHECK-I386:   iundefsym: 7
// CHECK-I386:   nundefsym: 3
// CHECK-I386:   tocoff: 0
// CHECK-I386:   ntoc: 0
// CHECK-I386:   modtaboff: 0
// CHECK-I386:   nmodtab: 0
// CHECK-I386:   extrefsymoff: 0
// CHECK-I386:   nextrefsyms: 0
// CHECK-I386:   indirectsymoff: 0
// CHECK-I386:   nindirectsyms: 0
// CHECK-I386:   extreloff: 0
// CHECK-I386:   nextrel: 0
// CHECK-I386:   locreloff: 0
// CHECK-I386:   nlocrel: 0
// CHECK-I386: }

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
// CHECK-X86_64:   SizeOfLoadCommands: 352
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
// CHECK-X86_64:     Size: 0x1
// CHECK-X86_64:     Offset: 384
// CHECK-X86_64:     Alignment: 0
// CHECK-X86_64:     RelocationOffset: 0x0
// CHECK-X86_64:     RelocationCount: 0
// CHECK-X86_64:     Type: 0x0
// CHECK-X86_64:     Attributes [ (0x800004)
// CHECK-X86_64:       PureInstructions (0x800000)
// CHECK-X86_64:       SomeInstructions (0x4)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Reserved1: 0x0
// CHECK-X86_64:     Reserved2: 0x0
// CHECK-X86_64:     Reserved3: 0x0
// CHECK-X86_64:     SectionData (
// CHECK-X86_64:       0000: C3                                   |.|
// CHECK-X86_64:     )
// CHECK-X86_64:   }
// CHECK-X86_64:   Section {
// CHECK-X86_64:     Index: 1
// CHECK-X86_64:     Name: __data (5F 5F 64 61 74 61 00 00 00 00 00 00 00 00 00 00)
// CHECK-X86_64:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK-X86_64:     Address: 0x1
// CHECK-X86_64:     Size: 0x38
// CHECK-X86_64:     Offset: 385
// CHECK-X86_64:     Alignment: 0
// CHECK-X86_64:     RelocationOffset: 0x1BC
// CHECK-X86_64:     RelocationCount: 9
// CHECK-X86_64:     Type: 0x0
// CHECK-X86_64:     Attributes [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Reserved1: 0x0
// CHECK-X86_64:     Reserved2: 0x0
// CHECK-X86_64:     Reserved3: 0x0
// CHECK-X86_64:     SectionData (
// CHECK-X86_64:       0000: 00000000 00000000 00000000 00000000  |................|
// CHECK-X86_64:       0010: 00000000 00000000 04000000 00000000  |................|
// CHECK-X86_64:       0020: 00000000 00000000 00000000 00000000  |................|
// CHECK-X86_64:       0030: CFFFFFFF FFFFFFFF                    |........|
// CHECK-X86_64:     )
// CHECK-X86_64:   }
// CHECK-X86_64: ]
// CHECK-X86_64: Relocations [
// CHECK-X86_64:   Section __data {
// CHECK-X86_64:     0x2C 0 2 1 X86_64_RELOC_UNSIGNED 0 g
// CHECK-X86_64:     0x28 0 2 1 X86_64_RELOC_UNSIGNED 0 f
// CHECK-X86_64:     0x24 0 2 1 X86_64_RELOC_UNSIGNED 0 d3
// CHECK-X86_64:     0x20 0 2 1 X86_64_RELOC_UNSIGNED 0 d{{$}}
// CHECK-X86_64:     0x1C 0 2 1 X86_64_RELOC_UNSIGNED 0 d{{$}}
// CHECK-X86_64:     0x18 0 2 1 X86_64_RELOC_UNSIGNED 0 a
// CHECK-X86_64:     0x14 0 2 1 X86_64_RELOC_UNSIGNED 0 e
// CHECK-X86_64:     0x10 0 2 1 X86_64_RELOC_UNSIGNED 0 b
// CHECK-X86_64:     0x8 0 2 1 X86_64_RELOC_UNSIGNED 0 b
// CHECK-X86_64:   }
// CHECK-X86_64: ]
// CHECK-X86_64: Symbols [
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: a (13)
// CHECK-X86_64:     Type: Section (0xE)
// CHECK-X86_64:     Section: __data (0x2)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x5
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: b (11)
// CHECK-X86_64:     Type: Section (0xE)
// CHECK-X86_64:     Section: __data (0x2)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x5
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: c (9)
// CHECK-X86_64:     Type: Section (0xE)
// CHECK-X86_64:     Section: __data (0x2)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x9
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: e (5)
// CHECK-X86_64:     Type: Section (0xE)
// CHECK-X86_64:     Section: __data (0x2)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x200)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x9
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: g (1)
// CHECK-X86_64:     Type: Section (0xE)
// CHECK-X86_64:     Section: __data (0x2)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0xD
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: f (3)
// CHECK-X86_64:     Type: Section (0xE)
// CHECK-X86_64:     Section: __data (0x2)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0xD
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: t0 (21)
// CHECK-X86_64:     Type: Section (0xE)
// CHECK-X86_64:     Section: __text (0x1)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x0
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: d (7)
// CHECK-X86_64:     Extern
// CHECK-X86_64:     Type: Undef (0x0)
// CHECK-X86_64:     Section:  (0x0)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x0
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: d2 (18)
// CHECK-X86_64:     Extern
// CHECK-X86_64:     Type: Indirect (0xA)
// CHECK-X86_64:     Section:  (0x0)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x0)
// CHECK-X86_64:     ]
// CHECK-X86_64:     Value: 0x7
// CHECK-X86_64:   }
// CHECK-X86_64:   Symbol {
// CHECK-X86_64:     Name: d3 (15)
// CHECK-X86_64:     Extern
// CHECK-X86_64:     Type: Undef (0x0)
// CHECK-X86_64:     Section:  (0x0)
// CHECK-X86_64:     RefType: UndefinedNonLazy (0x0)
// CHECK-X86_64:     Flags [ (0x200)
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
// CHECK-X86_64:   Size: 232
// CHECK-X86_64:   vmaddr: 0x0
// CHECK-X86_64:   vmsize: 0x39
// CHECK-X86_64:   fileoff: 384
// CHECK-X86_64:   filesize: 57
// CHECK-X86_64:   maxprot: rwx
// CHECK-X86_64:   initprot: rwx
// CHECK-X86_64:   nsects: 2
// CHECK-X86_64:   flags: 0x0
// CHECK-X86_64: }
// CHECK-X86_64: Dysymtab {
// CHECK-X86_64:   ilocalsym: 0
// CHECK-X86_64:   nlocalsym: 7
// CHECK-X86_64:   iextdefsym: 7
// CHECK-X86_64:   nextdefsym: 0
// CHECK-X86_64:   iundefsym: 7
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
