// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --file-headers -S --sd -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols | FileCheck %s

_text_a:
        xorl %eax,%eax
_text_b:
        xorl %eax,%eax
Ltext_c:
        xorl %eax,%eax
Ltext_d:
        xorl %eax,%eax

        movl $(_text_a - _text_b), %eax
Ltext_expr_0 = _text_a - _text_b
        movl $(Ltext_expr_0), %eax

        movl $(Ltext_c - _text_b), %eax
Ltext_expr_1 = Ltext_c - _text_b
        movl $(Ltext_expr_1), %eax

        movl $(Ltext_d - Ltext_c), %eax
Ltext_expr_2 = Ltext_d - Ltext_c
        movl $(Ltext_expr_2), %eax

        movl $(_text_a + Ltext_expr_0), %eax

        .data
_data_a:
        .long 0
_data_b:
        .long 0
Ldata_c:
        .long 0
Ldata_d:
        .long 0

        .long _data_a - _data_b
Ldata_expr_0 = _data_a - _data_b
        .long Ldata_expr_0

        .long Ldata_c - _data_b
Ldata_expr_1 = Ldata_c - _data_b
        .long Ldata_expr_1

        .long Ldata_d - Ldata_c
Ldata_expr_2 = Ldata_d - Ldata_c
        .long Ldata_expr_2

        .long _data_a + Ldata_expr_0

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
// CHECK:     Size: 0x2B
// CHECK:     Offset: 340
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x1AC
// CHECK:     RelocationCount: 3
// CHECK:     Type: 0x0
// CHECK:     Attributes [ (0x800004)
// CHECK:       PureInstructions (0x800000)
// CHECK:       SomeInstructions (0x4)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 31C031C0 31C031C0 B8FEFFFF FFB8FEFF  |1.1.1.1.........|
// CHECK:       0010: FFFFB802 000000B8 02000000 B8020000  |................|
// CHECK:       0020: 00B80200 0000B8FE FFFFFF             |...........|
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 1
// CHECK:     Name: __data (5F 5F 64 61 74 61 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x2B
// CHECK:     Size: 0x2C
// CHECK:     Offset: 383
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x1C4
// CHECK:     RelocationCount: 3
// CHECK:     Type: 0x0
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 00000000 00000000 00000000 00000000  |................|
// CHECK:       0010: FCFFFFFF FCFFFFFF 04000000 04000000  |................|
// CHECK:       0020: 04000000 04000000 27000000           |........'...|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK:   Section __text {
// CHECK:     0x27 0 2 n/a GENERIC_RELOC_VANILLA 1 0x0
// CHECK:     0x9 0 2 n/a GENERIC_RELOC_LOCAL_SECTDIFF 1 0x0
// CHECK:     0x0 0 2 n/a GENERIC_RELOC_PAIR 1 0x2
// CHECK:   }
// CHECK:   Section __data {
// CHECK:     0x28 0 2 n/a GENERIC_RELOC_VANILLA 1 0x2B
// CHECK:     0x10 0 2 n/a GENERIC_RELOC_LOCAL_SECTDIFF 1 0x2B
// CHECK:     0x0 0 2 n/a GENERIC_RELOC_PAIR 1 0x2F
// CHECK:   }
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: _text_a (17)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _text_b (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x2
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _data_a (25)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x2B
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _data_b (9)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x2F
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
// CHECK:   vmsize: 0x57
// CHECK:   fileoff: 340
// CHECK:   filesize: 87
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 2
// CHECK:   flags: 0x0
// CHECK: }
// CHECK: Dysymtab {
// CHECK:   ilocalsym: 0
// CHECK:   nlocalsym: 4
// CHECK:   iextdefsym: 4
// CHECK:   nextdefsym: 0
// CHECK:   iundefsym: 4
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
