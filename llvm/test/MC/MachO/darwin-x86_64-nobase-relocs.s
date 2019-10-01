// RUN: llvm-mc -n -triple x86_64-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --file-headers -S --sd -r --macho-segment | FileCheck %s

// Test case for rdar://10062261

// Must be no base, non-temporary, symbol before the reference to Lbar at the
// start of the section.  What we are testing for is that the reference does not
// create a relocation entry.
.text
Ladd:
	nop
	jmp Lbar
	.byte 0x0f,0x1f,0x40,0x00
	.byte 0x0f,0x1f,0x40,0x00
Lbar:	
	mov $1, %eax
	ret

// CHECK: File: <stdin>
// CHECK: Format: Mach-O 64-bit x86-64
// CHECK: Arch: x86_64
// CHECK: AddressSize: 64bit
// CHECK: MachHeader {
// CHECK:   Magic: Magic64 (0xFEEDFACF)
// CHECK:   CpuType: X86-64 (0x1000007)
// CHECK:   CpuSubType: CPU_SUBTYPE_X86_64_ALL (0x3)
// CHECK:   FileType: Relocatable (0x1)
// CHECK:   NumOfLoadCommands: 2
// CHECK:   SizeOfLoadCommands: 168
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
// CHECK:     Size: 0x11
// CHECK:     Offset: 200
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
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 90EB080F 1F40000F 1F4000B8 01000000  |.....@...@......|
// CHECK:       0010: C3                                   |.|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK: ]
// CHECK: Segment {
// CHECK:   Cmd: LC_SEGMENT_64
// CHECK:   Name: 
// CHECK:   Size: 152
// CHECK:   vmaddr: 0x0
// CHECK:   vmsize: 0x11
// CHECK:   fileoff: 200
// CHECK:   filesize: 17
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 1
// CHECK:   flags: 0x0
// CHECK: }
