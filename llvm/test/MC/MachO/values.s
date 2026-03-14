// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --file-headers -S -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols - | FileCheck %s

        .long 0
text_def_int:
        .long 0

        .globl text_def_ext
text_def_ext:
        .long 0

        .data
        .long 0
data_def_int:
        .long 0

        .globl data_def_ext
data_def_ext:
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
// CHECK:     Size: 0xC
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
// CHECK:     Address: 0xC
// CHECK:     Size: 0xC
// CHECK:     Offset: 352
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
// CHECK:     Name: text_def_int (27)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x4
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: data_def_int (40)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x10
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: data_def_ext (14)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x14
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: text_def_ext (1)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x8
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
// CHECK:   vmsize: 0x18
// CHECK:   fileoff: 340
// CHECK:   filesize: 24
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 2
// CHECK:   flags: 0x0
// CHECK: }
// CHECK: Dysymtab {
// CHECK:   ilocalsym: 0
// CHECK:   nlocalsym: 2
// CHECK:   iextdefsym: 2
// CHECK:   nextdefsym: 2
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
