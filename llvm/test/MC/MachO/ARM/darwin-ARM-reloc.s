@ RUN: llvm-mc -n -triple armv7-apple-darwin10 %s -filetype=obj -o %t.obj
@ RUN: llvm-readobj -file-headers -s -sd -r -t -macho-segment -macho-dysymtab -macho-indirect-symbols < %t.obj > %t.dump
@ RUN: FileCheck < %t.dump %s

	.syntax unified
        .text
_f0:
        bl _printf

_f1:
        bl _f0

        .data
_d0:
Ld0_0:
        .long Lsc0_0 - Ld0_0

	.section	__TEXT,__cstring,cstring_literals
Lsc0_0:
        .long 0

        .subsections_via_symbols

@ CHECK: File: <stdin>
@ CHECK: Format: Mach-O arm
@ CHECK: Arch: arm
@ CHECK: AddressSize: 32bit
@ CHECK: MachHeader {
@ CHECK:   Magic: Magic (0xFEEDFACE)
@ CHECK:   CpuType: Arm (0xC)
@ CHECK:   CpuSubType: CPU_SUBTYPE_ARM_V7 (0x9)
@ CHECK:   FileType: Relocatable (0x1)
@ CHECK:   NumOfLoadCommands: 4
@ CHECK:   SizeOfLoadCommands: 380
@ CHECK:   Flags [ (0x2000)
@ CHECK:     MH_SUBSECTIONS_VIA_SYMBOLS (0x2000)
@ CHECK:   ]
@ CHECK: }
@ CHECK: Sections [
@ CHECK:   Section {
@ CHECK:     Index: 0
@ CHECK:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
@ CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
@ CHECK:     Address: 0x0
@ CHECK:     Size: 0x8
@ CHECK:     Offset: 408
@ CHECK:     Alignment: 0
@ CHECK:     RelocationOffset: 0x1A8
@ CHECK:     RelocationCount: 2
@ CHECK:     Type: 0x0
@ CHECK:     Attributes [ (0x800004)
@ CHECK:       PureInstructions (0x800000)
@ CHECK:       SomeInstructions (0x4)
@ CHECK:     ]
@ CHECK:     Reserved1: 0x0
@ CHECK:     Reserved2: 0x0
@ CHECK:     SectionData (
@ CHECK:       0000: FEFFFFEB FDFFFFEB                    |........|
@ CHECK:     )
@ CHECK:   }
@ CHECK:   Section {
@ CHECK:     Index: 1
@ CHECK:     Name: __data (5F 5F 64 61 74 61 00 00 00 00 00 00 00 00 00 00)
@ CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
@ CHECK:     Address: 0x8
@ CHECK:     Size: 0x4
@ CHECK:     Offset: 416
@ CHECK:     Alignment: 0
@ CHECK:     RelocationOffset: 0x1B8
@ CHECK:     RelocationCount: 2
@ CHECK:     Type: 0x0
@ CHECK:     Attributes [ (0x0)
@ CHECK:     ]
@ CHECK:     Reserved1: 0x0
@ CHECK:     Reserved2: 0x0
@ CHECK:     SectionData (
@ CHECK:       0000: 04000000                             |....|
@ CHECK:     )
@ CHECK:   }
@ CHECK:   Section {
@ CHECK:     Index: 2
@ CHECK:     Name: __cstring (5F 5F 63 73 74 72 69 6E 67 00 00 00 00 00 00 00)
@ CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
@ CHECK:     Address: 0xC
@ CHECK:     Size: 0x4
@ CHECK:     Offset: 420
@ CHECK:     Alignment: 0
@ CHECK:     RelocationOffset: 0x0
@ CHECK:     RelocationCount: 0
@ CHECK:     Type: ExtReloc (0x2)
@ CHECK:     Attributes [ (0x0)
@ CHECK:     ]
@ CHECK:     Reserved1: 0x0
@ CHECK:     Reserved2: 0x0
@ CHECK:     SectionData (
@ CHECK:       0000: 00000000                             |....|
@ CHECK:     )
@ CHECK:   }
@ CHECK: ]
@ CHECK: Relocations [
@ CHECK:   Section __text {
@ CHECK:     0x4 1 2 0 ARM_RELOC_BR24 0 __text
@ CHECK:     0x0 1 2 1 ARM_RELOC_BR24 0 _printf
@ CHECK:   }
@ CHECK:   Section __data {
@ CHECK:     0x0 0 2 n/a ARM_RELOC_SECTDIFF 1 0xC
@ CHECK:     0x0 0 2 n/a ARM_RELOC_PAIR 1 0x8
@ CHECK:   }
@ CHECK: ]
@ CHECK: Symbols [
@ CHECK:   Symbol {
@ CHECK:     Name: _f0 (13)
@ CHECK:     Type: Section (0xE)
@ CHECK:     Section: __text (0x1)
@ CHECK:     RefType: UndefinedNonLazy (0x0)
@ CHECK:     Flags [ (0x0)
@ CHECK:     ]
@ CHECK:     Value: 0x0
@ CHECK:   }
@ CHECK:   Symbol {
@ CHECK:     Name: _f1 (9)
@ CHECK:     Type: Section (0xE)
@ CHECK:     Section: __text (0x1)
@ CHECK:     RefType: UndefinedNonLazy (0x0)
@ CHECK:     Flags [ (0x0)
@ CHECK:     ]
@ CHECK:     Value: 0x4
@ CHECK:   }
@ CHECK:   Symbol {
@ CHECK:     Name: _d0 (17)
@ CHECK:     Type: Section (0xE)
@ CHECK:     Section: __data (0x2)
@ CHECK:     RefType: UndefinedNonLazy (0x0)
@ CHECK:     Flags [ (0x0)
@ CHECK:     ]
@ CHECK:     Value: 0x8
@ CHECK:   }
@ CHECK:   Symbol {
@ CHECK:     Name: _printf (1)
@ CHECK:     Extern
@ CHECK:     Type: Undef (0x0)
@ CHECK:     Section:  (0x0)
@ CHECK:     RefType: UndefinedNonLazy (0x0)
@ CHECK:     Flags [ (0x0)
@ CHECK:     ]
@ CHECK:     Value: 0x0
@ CHECK:   }
@ CHECK: ]
@ CHECK: Indirect Symbols {
@ CHECK:   Number: 0
@ CHECK:   Symbols [
@ CHECK:   ]
@ CHECK: }
@ CHECK: Segment {
@ CHECK:   Cmd: LC_SEGMENT
@ CHECK:   Name: 
@ CHECK:   Size: 260
@ CHECK:   vmaddr: 0x0
@ CHECK:   vmsize: 0x10
@ CHECK:   fileoff: 408
@ CHECK:   filesize: 16
@ CHECK:   maxprot: rwx
@ CHECK:   initprot: rwx
@ CHECK:   nsects: 3
@ CHECK:   flags: 0x0
@ CHECK: }
@ CHECK: Dysymtab {
@ CHECK:   ilocalsym: 0
@ CHECK:   nlocalsym: 3
@ CHECK:   iextdefsym: 3
@ CHECK:   nextdefsym: 0
@ CHECK:   iundefsym: 3
@ CHECK:   nundefsym: 1
@ CHECK:   tocoff: 0
@ CHECK:   ntoc: 0
@ CHECK:   modtaboff: 0
@ CHECK:   nmodtab: 0
@ CHECK:   extrefsymoff: 0
@ CHECK:   nextrefsyms: 0
@ CHECK:   indirectsymoff: 0
@ CHECK:   nindirectsyms: 0
@ CHECK:   extreloff: 0
@ CHECK:   nextrel: 0
@ CHECK:   locreloff: 0
@ CHECK:   nlocrel: 0
@ CHECK: }
