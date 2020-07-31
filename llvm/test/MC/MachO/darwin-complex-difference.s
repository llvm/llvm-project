// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o %t.o
// RUN: llvm-readobj --file-headers -S --sd -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols < %t.o - > %t.dump
// RUN: FileCheck < %t.dump %s
        
_a:
L0:     
        .long 1
L1:     
        .long 2
        .long _c - _d + 4
        .long (_c - L0) - (_d - L1) // == (_c - _d) + (L1 - L0)
                                    // == (_c - _d + 4)
_c:
        .long 0
_d:
        .long 0

// CHECK: File: <stdin>
// CHECK: Format: Mach-O 64-bit x86-64
// CHECK: Arch: x86_64
// CHECK: AddressSize: 64bit
// CHECK: MachHeader {
// CHECK:   Magic: Magic64 (0xFEEDFACF)
// CHECK:   CpuType: X86-64 (0x1000007)
// CHECK:   CpuSubType: CPU_SUBTYPE_X86_64_ALL (0x3)
// CHECK:   FileType: Relocatable (0x1)
// CHECK:   NumOfLoadCommands: 4
// CHECK:   SizeOfLoadCommands: 272
// CHECK:   Flags [ (0x0)
// CHECK:   ]
// CHECK:   Reserved: 0x0
// CHECK: }
// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Index: 0
// CHECK:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x18
// CHECK:     Offset: 304
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x148
// CHECK:     RelocationCount: 4
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x800000)
// CHECK:       PureInstructions (0x800000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 01000000 02000000 04000000 04000000  |................|
// CHECK:       0010: 00000000 00000000                    |........|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK:   Section __text {
// CHECK:     0xC 0 2 1 X86_64_RELOC_SUBTRACTOR 0 _d
// CHECK:     0xC 0 2 1 X86_64_RELOC_UNSIGNED 0 _c
// CHECK:     0x8 0 2 1 X86_64_RELOC_SUBTRACTOR 0 _d
// CHECK:     0x8 0 2 1 X86_64_RELOC_UNSIGNED 0 _c
// CHECK:   }
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: _a (7)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _c (4)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x10
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _d (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x14
// CHECK:   }
// CHECK: ]
// CHECK: Indirect Symbols {
// CHECK:   Number: 0
// CHECK:   Symbols [
// CHECK:   ]
// CHECK: }
// CHECK: Segment {
// CHECK:   Cmd: LC_SEGMENT_64
// CHECK:   Name: 
// CHECK:   Size: 152
// CHECK:   vmaddr: 0x0
// CHECK:   vmsize: 0x18
// CHECK:   fileoff: 304
// CHECK:   filesize: 24
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 1
// CHECK:   flags: 0x0
// CHECK: }
// CHECK: Dysymtab {
// CHECK:   ilocalsym: 0
// CHECK:   nlocalsym: 3
// CHECK:   iextdefsym: 3
// CHECK:   nextdefsym: 0
// CHECK:   iundefsym: 3
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
