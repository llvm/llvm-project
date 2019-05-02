// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --file-headers -S -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols | FileCheck %s

direction_labels:
10:     nop
        jmp 10b
        nop
	jne 0f
0:	nop
	jne 0b
        jmp 11f
11:     nop
        ret

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
// CHECK:   SizeOfLoadCommands: 244
// CHECK:   Flags [ (0x0)
// CHECK:   ]
// CHECK: }
// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Index: 0
// CHECK:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0xD
// CHECK:     Offset: 272
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: 0x0
// CHECK:     Attributes [ (0x800004)
// CHECK:       PureInstructions (0x800000)
// CHECK:       SomeInstructions (0x4)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: direction_labels (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
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
// CHECK:   Size: 124
// CHECK:   vmaddr: 0x0
// CHECK:   vmsize: 0xD
// CHECK:   fileoff: 272
// CHECK:   filesize: 13
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 1
// CHECK:   flags: 0x0
// CHECK: }
// CHECK: Dysymtab {
// CHECK:   ilocalsym: 0
// CHECK:   nlocalsym: 1
// CHECK:   iextdefsym: 1
// CHECK:   nextdefsym: 0
// CHECK:   iundefsym: 1
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
