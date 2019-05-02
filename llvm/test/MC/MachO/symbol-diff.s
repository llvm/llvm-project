// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj --file-headers -S --sd -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols | FileCheck %s
_g:
LFB2:
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
_g.eh:
	.quad	LFB2-.

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
// CHECK:   SizeOfLoadCommands: 352
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
// CHECK:     Size: 0x0
// CHECK:     Offset: 384
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: 0x0
// CHECK:     Attributes [ (0x800000)
// CHECK:       PureInstructions (0x800000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 1
// CHECK:     Name: __eh_frame (5F 5F 65 68 5F 66 72 61 6D 65 00 00 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x0
// CHECK:     Size: 0x8
// CHECK:     Offset: 384
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x188
// CHECK:     RelocationCount: 2
// CHECK:     Type: 0xB
// CHECK:     Attributes [ (0x680000)
// CHECK:       LiveSupport (0x80000)
// CHECK:       NoTOC (0x400000)
// CHECK:       StripStaticSyms (0x200000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 00000000 00000000                    |........|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK:   Section __eh_frame {
// CHECK:     0x0 0 3 1 X86_64_RELOC_SUBTRACTOR 0 _g.eh
// CHECK:     0x0 0 3 1 X86_64_RELOC_UNSIGNED 0 _g
// CHECK:   }
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: _g (7)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _g.eh (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __eh_frame (0x2)
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
// CHECK:   Cmd: LC_SEGMENT_64
// CHECK:   Name: 
// CHECK:   Size: 232
// CHECK:   vmaddr: 0x0
// CHECK:   vmsize: 0x8
// CHECK:   fileoff: 384
// CHECK:   filesize: 8
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 2
// CHECK:   flags: 0x0
// CHECK: }
// CHECK: Dysymtab {
// CHECK:   ilocalsym: 0
// CHECK:   nlocalsym: 2
// CHECK:   iextdefsym: 2
// CHECK:   nextdefsym: 0
// CHECK:   iundefsym: 2
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
