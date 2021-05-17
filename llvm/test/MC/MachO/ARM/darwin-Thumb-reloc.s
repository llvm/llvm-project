@ RUN: llvm-mc -n -triple thumbv7-apple-darwin10 %s -filetype=obj -o %t.obj
@ RUN: llvm-readobj --file-headers -S --sd -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols - < %t.obj > %t.dump
@ RUN: FileCheck < %t.dump %s

	.syntax unified
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.align	2
	.code	16
	.thumb_func	_main
_main:
LPC0_0:
	blx	_printf
	.align	2
LCPI0_0:
	.long	L_.str-(LPC0_0+4)

	.section	__TEXT,__cstring,cstring_literals
	.align	2
L_.str:
	.asciz	 "s0"

.subsections_via_symbols

@ CHECK: File: <stdin>
@ CHECK: Format: Mach-O arm
@ CHECK: Arch: arm
@ CHECK: AddressSize: 32bit
@ CHECK: MachHeader {
@ CHECK:   Magic: Magic (0xFEEDFACE)
@ CHECK:   CpuType: Arm (0xC)
@ CHECK:   CpuSubType: CPU_SUBTYPE_ARM_V7 (0x9)
@ CHECK:   FileType: Relocatable (0x1)
@ CHECK:   NumOfLoadCommands: 4
@ CHECK:   SizeOfLoadCommands: 312
@ CHECK:   Flags [ (0x2000)
@ CHECK:     MH_SUBSECTIONS_VIA_SYMBOLS (0x2000)
@ CHECK:   ]
@ CHECK: }
@ CHECK: Sections [
@ CHECK:   Section {
@ CHECK:     Index: 0
@ CHECK:     Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
@ CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
@ CHECK:     Address: 0x0
@ CHECK:     Size: 0x8
@ CHECK:     Offset: 340
@ CHECK:     Alignment: 2
@ CHECK:     RelocationOffset: 0x160
@ CHECK:     RelocationCount: 3
@ CHECK:     Type: Regular (0x0)
@ CHECK:     Attributes [ (0x800004)
@ CHECK:       PureInstructions (0x800000)
@ CHECK:       SomeInstructions (0x4)
@ CHECK:     ]
@ CHECK:     Reserved1: 0x0
@ CHECK:     Reserved2: 0x0
@ CHECK:     SectionData (
@ CHECK:       0000: FFF7FEEF 04000000                    |........|
@ CHECK:     )
@ CHECK:   }
@ CHECK:   Section {
@ CHECK:     Index: 1
@ CHECK:     Name: __cstring (5F 5F 63 73 74 72 69 6E 67 00 00 00 00 00 00 00)
@ CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
@ CHECK:     Address: 0x8
@ CHECK:     Size: 0x3
@ CHECK:     Offset: 348
@ CHECK:     Alignment: 2
@ CHECK:     RelocationOffset: 0x0
@ CHECK:     RelocationCount: 0
@ CHECK:     Type: CStringLiterals (0x2)
@ CHECK:     Attributes [ (0x0)
@ CHECK:     ]
@ CHECK:     Reserved1: 0x0
@ CHECK:     Reserved2: 0x0
@ CHECK:     SectionData (
@ CHECK:       0000: 733000                               |s0.|
@ CHECK:     )
@ CHECK:   }
@ CHECK: ]
@ CHECK: Relocations [
@ CHECK:   Section __text {
@ CHECK:     0x4 0 2 n/a ARM_RELOC_SECTDIFF 1 0x8
@ CHECK:     0x0 0 2 n/a ARM_RELOC_PAIR 1 0x0
@ CHECK:     0x0 1 2 1 ARM_THUMB_RELOC_BR22 0 _printf
@ CHECK:   }
@ CHECK: ]
@ CHECK: Symbols [
@ CHECK:   Symbol {
@ CHECK:     Name: _main (1)
@ CHECK:     Extern
@ CHECK:     Type: Section (0xE)
@ CHECK:     Section: __text (0x1)
@ CHECK:     RefType: UndefinedNonLazy (0x0)
@ CHECK:     Flags [ (0x8)
@ CHECK:       ThumbDef (0x8)
@ CHECK:     ]
@ CHECK:     Value: 0x0
@ CHECK:   }
@ CHECK:   Symbol {
@ CHECK:     Name: _printf (7)
@ CHECK:     Extern
@ CHECK:     Type: Undef (0x0)
@ CHECK:     Section:  (0x0)
@ CHECK:     RefType: UndefinedNonLazy (0x0)
@ CHECK:     Flags [ (0x0)
@ CHECK:     ]
@ CHECK:     Value: 0x0
@ CHECK:   }
@ CHECK: ]
@ CHECK: Indirect Symbols {
@ CHECK:   Number: 0
@ CHECK:   Symbols [
@ CHECK:   ]
@ CHECK: }
@ CHECK: Segment {
@ CHECK:   Cmd: LC_SEGMENT
@ CHECK:   Name: 
@ CHECK:   Size: 192
@ CHECK:   vmaddr: 0x0
@ CHECK:   vmsize: 0xB
@ CHECK:   fileoff: 340
@ CHECK:   filesize: 11
@ CHECK:   maxprot: rwx
@ CHECK:   initprot: rwx
@ CHECK:   nsects: 2
@ CHECK:   flags: 0x0
@ CHECK: }
@ CHECK: Dysymtab {
@ CHECK:   ilocalsym: 0
@ CHECK:   nlocalsym: 0
@ CHECK:   iextdefsym: 0
@ CHECK:   nextdefsym: 1
@ CHECK:   iundefsym: 1
@ CHECK:   nundefsym: 1
@ CHECK:   tocoff: 0
@ CHECK:   ntoc: 0
@ CHECK:   modtaboff: 0
@ CHECK:   nmodtab: 0
@ CHECK:   extrefsymoff: 0
@ CHECK:   nextrefsyms: 0
@ CHECK:   indirectsymoff: 0
@ CHECK:   nindirectsyms: 0
@ CHECK:   extreloff: 0
@ CHECK:   nextrel: 0
@ CHECK:   locreloff: 0
@ CHECK:   nlocrel: 0
@ CHECK: }
