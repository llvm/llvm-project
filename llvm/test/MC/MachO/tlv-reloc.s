// RUN: llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o - | llvm-readobj --file-headers -S --sd -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols | FileCheck %s

.tdata
_a$tlv$init:
	.long 4


.tlv
	.globl _a
_a:
	.quad __tlv_bootstrap
	.quad 0
	.quad _a$tlv$init

.text
	.globl _foo
	.align 4, 0x90

_foo:
	 movq   _a@TLVP(%rip), %rdi
	 call	*(%rdi) # returns &a in %rax
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
// CHECK:   NumOfLoadCommands: 3
// CHECK:   SizeOfLoadCommands: 416
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
// CHECK:     Size: 0xA
// CHECK:     Offset: 448
// CHECK:     Alignment: 4
// CHECK:     RelocationOffset: 0x1E8
// CHECK:     RelocationCount: 1
// CHECK:     Type: 0x0
// CHECK:     Attributes [ (0x800004)
// CHECK:       PureInstructions (0x800000)
// CHECK:       SomeInstructions (0x4)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 488B3D00 000000FF 17C3               |H.=.......|
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 1
// CHECK:     Name: __thread_data (5F 5F 74 68 72 65 61 64 5F 64 61 74 61 00 00 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0xA
// CHECK:     Size: 0x4
// CHECK:     Offset: 458
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: 0x11
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 04000000                             |....|
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 2
// CHECK:     Name: __thread_vars (5F 5F 74 68 72 65 61 64 5F 76 61 72 73 00 00 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0xE
// CHECK:     Size: 0x18
// CHECK:     Offset: 462
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x1F0
// CHECK:     RelocationCount: 2
// CHECK:     Type: 0x13
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 00000000 00000000 00000000 00000000  |................|
// CHECK:       0010: 00000000 00000000                    |........|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK:   Section __text {
// CHECK:     0x3 1 2 1 X86_64_RELOC_TLV 0 _a
// CHECK:   }
// CHECK:   Section __thread_vars {
// CHECK:     0x10 0 3 1 X86_64_RELOC_UNSIGNED 0 _a$tlv$init
// CHECK:     0x0 0 3 1 X86_64_RELOC_UNSIGNED 0 __tlv_bootstrap
// CHECK:   }
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: _a$tlv$init (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __thread_data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0xA
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _a (34)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __thread_vars (0x3)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0xE
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _foo (29)
// CHECK:     Extern
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: __tlv_bootstrap (13)
// CHECK:     Extern
// CHECK:     Type: Undef (0x0)
// CHECK:     Section:  (0x0)
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
// CHECK:   Size: 312
// CHECK:   vmaddr: 0x0
// CHECK:   vmsize: 0x26
// CHECK:   fileoff: 448
// CHECK:   filesize: 38
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 3
// CHECK:   flags: 0x0
// CHECK: }
// CHECK: Dysymtab {
// CHECK:   ilocalsym: 0
// CHECK:   nlocalsym: 1
// CHECK:   iextdefsym: 1
// CHECK:   nextdefsym: 2
// CHECK:   iundefsym: 3
// CHECK:   nundefsym: 1
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
