// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --file-headers -S --sd -r --macho-segment | FileCheck %s

   ja 1f
1: nop
   jae 1f
1: nop
   jb 1f
1: nop
   jbe 1f
1: nop
   jc 1f
1: nop
   jecxz 1f
1: nop
   jecxz 1f
1: nop
   je 1f
1: nop
   jg 1f
1: nop
   jge 1f
1: nop
   jl 1f
1: nop
   jle 1f
1: nop
   jna 1f
1: nop
   jnae 1f
1: nop
   jnb 1f
1: nop
   jnbe 1f
1: nop
   jnc 1f
1: nop
   jne 1f
1: nop
   jng 1f
1: nop
   jnge 1f
1: nop
   jnl 1f
1: nop
   jnle 1f
1: nop
   jno 1f
1: nop
   jnp 1f
1: nop
   jns 1f
1: nop
   jnz 1f
1: nop
   jo 1f
1: nop
   jp 1f
1: nop
   jpe 1f
1: nop
   jpo 1f
1: nop
   js 1f
1: nop
   jz 1f
1: nop

// CHECK: File: <stdin>
// CHECK: Format: Mach-O 32-bit i386
// CHECK: Arch: i386
// CHECK: AddressSize: 32bit
// CHECK: MachHeader {
// CHECK:   Magic: Magic (0xFEEDFACE)
// CHECK:   CpuType: X86 (0x7)
// CHECK:   CpuSubType: CPU_SUBTYPE_I386_ALL (0x3)
// CHECK:   FileType: Relocatable (0x1)
// CHECK:   NumOfLoadCommands: 2
// CHECK:   SizeOfLoadCommands: 140
// CHECK:   Flags [ (0x0)
// CHECK:   ]
// CHECK: }
// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Index: 0
// CHECK:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x60
// CHECK:     Offset: 168
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
// CHECK:     SectionData (
// CHECK:       0000: 77009073 00907200 90760090 720090E3  |w..s..r..v..r...|
// CHECK:       0010: 0090E300 90740090 7F00907D 00907C00  |.....t.....}..|.|
// CHECK:       0020: 907E0090 76009072 00907300 90770090  |.~..v..r..s..w..|
// CHECK:       0030: 73009075 00907E00 907C0090 7D00907F  |s..u..~..|..}...|
// CHECK:       0040: 00907100 907B0090 79009075 00907000  |..q..{..y..u..p.|
// CHECK:       0050: 907A0090 7A00907B 00907800 90740090  |.z..z..{..x..t..|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK: ]
// CHECK: Segment {
// CHECK:   Cmd: LC_SEGMENT
// CHECK:   Name: 
// CHECK:   Size: 124
// CHECK:   vmaddr: 0x0
// CHECK:   vmsize: 0x60
// CHECK:   fileoff: 168
// CHECK:   filesize: 96
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 1
// CHECK:   flags: 0x0
// CHECK: }
