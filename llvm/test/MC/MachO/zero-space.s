// RUN: llvm-mc -triple x86_64-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -file-headers -s -sd -r -t --macho-segment --macho-dysymtab --macho-indirect-symbols | FileCheck %s

        .const
        .p2align 6
        Lzero:
          .space 64
          .zero 64

// CHECK: File: <stdin>
// CHECK-NEXT: Format: Mach-O 64-bit x86-64
// CHECK-NEXT: Arch: x86_64
// CHECK-NEXT: AddressSize: 64bit
// CHECK-NEXT: MachHeader {
// CHECK-NEXT:   Magic: Magic64 (0xFEEDFACF)
// CHECK-NEXT:   CpuType: X86-64 (0x1000007)
// CHECK-NEXT:   CpuSubType: CPU_SUBTYPE_X86_64_ALL (0x3)
// CHECK-NEXT:   FileType: Relocatable (0x1)
// CHECK-NEXT:   NumOfLoadCommands: 2
// CHECK-NEXT:   SizeOfLoadCommands: 248
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Reserved: 0x0
// CHECK-NEXT: }
// CHECK-NEXT: Sections [
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 0
// CHECK-NEXT:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 280
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x800000)
// CHECK-NEXT:       PureInstructions (0x800000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:     Reserved3: 0x0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 1
// CHECK-NEXT:     Name: __const (5F 5F 63 6F 6E 73 74 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x80
// CHECK-NEXT:     Offset: 280
// CHECK-NEXT:     Alignment: 6
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:     Reserved3: 0x0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 00000000 00000000 00000000 00000000  |................|
// CHECK-NEXT:       0010: 00000000 00000000 00000000 00000000  |................|
// CHECK-NEXT:       0020: 00000000 00000000 00000000 00000000  |................|
// CHECK-NEXT:       0030: 00000000 00000000 00000000 00000000  |................|
// CHECK-NEXT:       0040: 00000000 00000000 00000000 00000000  |................|
// CHECK-NEXT:       0050: 00000000 00000000 00000000 00000000  |................|
// CHECK-NEXT:       0060: 00000000 00000000 00000000 00000000  |................|
// CHECK-NEXT:       0070: 00000000 00000000 00000000 00000000  |................|
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK-NEXT: Relocations [
// CHECK-NEXT: ]
// CHECK-NEXT: Symbols [
// CHECK-NEXT: ]
// CHECK-NEXT: Segment {
// CHECK-NEXT:   Cmd: LC_SEGMENT_64
// CHECK-NEXT:   Name:
// CHECK-NEXT:   Size: 232
// CHECK-NEXT:   vmaddr: 0x0
// CHECK-NEXT:   vmsize: 0x80
// CHECK-NEXT:   fileoff: 280
// CHECK-NEXT:   filesize: 128
// CHECK-NEXT:   maxprot: rwx
// CHECK-NEXT:   initprot: rwx
// CHECK-NEXT:   nsects: 2
// CHECK-NEXT:   flags: 0x0
// CHECK-NEXT: }
