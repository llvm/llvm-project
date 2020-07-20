// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --file-headers -S --sd -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols - | FileCheck %s

	.section	__DATA,__datacoal_nt,coalesced
	.section	__TEXT,__const_coal,coalesced
	.globl	__ZTS3optIbE            ## @_ZTS3optIbE
	.weak_definition	__ZTS3optIbE
__ZTS3optIbE:


	.section	__DATA,__datacoal_nt,coalesced
	.globl	__ZTI3optIbE            ## @_ZTI3optIbE
	.weak_definition	__ZTI3optIbE

__ZTI3optIbE:
	.long	__ZTS3optIbE

// CHECK: File: <stdin>
// CHECK-NEXT: Format: Mach-O 32-bit i386
// CHECK-NEXT: Arch: i386
// CHECK-NEXT: AddressSize: 32bit
// CHECK-NEXT: MachHeader {
// CHECK-NEXT:   Magic: Magic (0xFEEDFACE)
// CHECK-NEXT:   CpuType: X86 (0x7)
// CHECK-NEXT:   CpuSubType: CPU_SUBTYPE_I386_ALL (0x3)
// CHECK-NEXT:   FileType: Relocatable (0x1)
// CHECK-NEXT:   NumOfLoadCommands: 4
// CHECK-NEXT:   SizeOfLoadCommands: 380
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
// CHECK-NEXT:     Offset: 408
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: Regular (0x0)
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
// CHECK-NEXT:     Name: __datacoal_nt (5F 5F 64 61 74 61 63 6F 61 6C 5F 6E 74 00 00 00)
// CHECK-NEXT:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Size: 0x4
// CHECK-NEXT:     Offset: 408
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x19C
// CHECK-NEXT:     RelocationCount: 1
// CHECK-NEXT:     Type: Coalesced (0xB)
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 00000000                             |....|
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 2
// CHECK-NEXT:     Name: __const_coal (5F 5F 63 6F 6E 73 74 5F 63 6F 61 6C 00 00 00 00)
// CHECK-NEXT:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:     Address: 0x4
// CHECK-NEXT:     Size: 0x0
// CHECK-NEXT:     Offset: 412
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:     RelocationOffset: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     Type: Coalesced (0xB)
// CHECK-NEXT:     Attributes [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Reserved1: 0x0
// CHECK-NEXT:     Reserved2: 0x0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK-NEXT: Relocations [
// CHECK-NEXT:   Section __datacoal_nt {
// CHECK-NEXT:     0x0 0 2 1 GENERIC_RELOC_VANILLA 0 __ZTS3optIbE
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK-NEXT: Symbols [
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: __ZTI3optIbE (14)
// CHECK-NEXT:     Extern
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __datacoal_nt (0x2)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x80)
// CHECK-NEXT:       WeakDef (0x80)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: __ZTS3optIbE (1)
// CHECK-NEXT:     Extern
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __const_coal (0x3)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x80)
// CHECK-NEXT:       WeakDef (0x80)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x4
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK-NEXT: Indirect Symbols {
// CHECK-NEXT:   Number: 0
// CHECK-NEXT:   Symbols [
// CHECK-NEXT:   ]
// CHECK-NEXT: }
// CHECK-NEXT: Segment {
// CHECK-NEXT:   Cmd: LC_SEGMENT
// CHECK-NEXT:   Name: 
// CHECK-NEXT:   Size: 260
// CHECK-NEXT:   vmaddr: 0x0
// CHECK-NEXT:   vmsize: 0x4
// CHECK-NEXT:   fileoff: 408
// CHECK-NEXT:   filesize: 4
// CHECK-NEXT:   maxprot: rwx
// CHECK-NEXT:   initprot: rwx
// CHECK-NEXT:   nsects: 3
// CHECK-NEXT:   flags: 0x0
// CHECK-NEXT: }
// CHECK-NEXT: Dysymtab {
// CHECK-NEXT:   ilocalsym: 0
// CHECK-NEXT:   nlocalsym: 0
// CHECK-NEXT:   iextdefsym: 0
// CHECK-NEXT:   nextdefsym: 2
// CHECK-NEXT:   iundefsym: 2
// CHECK-NEXT:   nundefsym: 0
// CHECK-NEXT:   tocoff: 0
// CHECK-NEXT:   ntoc: 0
// CHECK-NEXT:   modtaboff: 0
// CHECK-NEXT:   nmodtab: 0
// CHECK-NEXT:   extrefsymoff: 0
// CHECK-NEXT:   nextrefsyms: 0
// CHECK-NEXT:   indirectsymoff: 0
// CHECK-NEXT:   nindirectsyms: 0
// CHECK-NEXT:   extreloff: 0
// CHECK-NEXT:   nextrel: 0
// CHECK-NEXT:   locreloff: 0
// CHECK-NEXT:   nlocrel: 0
// CHECK-NEXT: }
