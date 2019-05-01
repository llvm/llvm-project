// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --file-headers -S --sd -r --macho-segment --macho-version-min | FileCheck %s

	.section	__TEXT,__text,regular,pure_instructions
Leh_func_begin0:
	.section	__TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
Ltmp3:
Ltmp4 = Leh_func_begin0-Ltmp3
	.long	Ltmp4

// CHECK: File: <stdin>
// CHECK-NEXT: Format: Mach-O 32-bit i386
// CHECK-NEXT: Arch: i386
// CHECK-NEXT: AddressSize: 32bit
// CHECK-NEXT: MachHeader {
// CHECK-NEXT:   Magic: Magic (0xFEEDFACE)
// CHECK-NEXT:   CpuType: X86 (0x7)
// CHECK-NEXT:   CpuSubType: CPU_SUBTYPE_I386_ALL (0x3)
// CHECK-NEXT:   FileType: Relocatable (0x1)
// CHECK-NEXT:   NumOfLoadCommands: 2
// CHECK-NEXT:   SizeOfLoadCommands: 208
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT: }
// CHECK-NEXT: Sections [
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 0
// CHECK-NEXT:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 236
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0x0
// CHECK-NEXT:     Attributes [ (0x800000)
// CHECK-NEXT:       PureInstructions (0x800000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 1
// CHECK-NEXT:     Name: __eh_frame (5F 5F 65 68 5F 66 72 61 6D 65 00 00 00 00 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x4
// CHECK-NEXT:     Offset: 236
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: 0xB
// CHECK-NEXT:     Attributes [ (0x680000)
// CHECK-NEXT:       LiveSupport (0x80000)
// CHECK-NEXT:       NoTOC (0x400000)
// CHECK-NEXT:       StripStaticSyms (0x200000)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 00000000                             |....|
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK-NEXT: Relocations [
// CHECK-NEXT: ]
// CHECK-NEXT: Segment {
// CHECK-NEXT:   Cmd: LC_SEGMENT
// CHECK-NEXT:   Name: 
// CHECK-NEXT:   Size: 192
// CHECK-NEXT:   vmaddr: 0x0
// CHECK-NEXT:   vmsize: 0x4
// CHECK-NEXT:   fileoff: 236
// CHECK-NEXT:   filesize: 4
// CHECK-NEXT:   maxprot: rwx
// CHECK-NEXT:   initprot: rwx
// CHECK-NEXT:   nsects: 2
// CHECK-NEXT:   flags: 0x0
// CHECK-NEXT: }
// CHECK-NEXT: MinVersion {
// CHECK-NEXT:   Cmd: LC_VERSION_MIN_MACOSX
// CHECK-NEXT:   Size: 16
// CHECK-NEXT:   Version: 10.5
// CHECK-NEXT:   SDK: n/a
// CHECK-NEXT: }
