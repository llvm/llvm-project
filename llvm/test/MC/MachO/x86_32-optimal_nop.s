// RUN: llvm-mc -triple i386-apple-darwin9 -mcpu=pentiumpro %s -filetype=obj -o - | llvm-readobj --file-headers -S --sd -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols - | FileCheck %s

# 1 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nop
        # 0x90
        .align 1, 0x90
        ret
# 2 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        # xchg %ax,%ax
        # 0x66, 0x90
        .align 2, 0x90
        ret
# 3 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nopl (%[re]ax)
        # 0x0f, 0x1f, 0x00
        .align 2, 0x90
        ret
# 4 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        # nopl 0(%[re]ax)
        # 0x0f, 0x1f, 0x40, 0x00
        .align 3, 0x90
        ret
# 5 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        # nopl 0(%[re]ax,%[re]ax,1)
        # 0x0f, 0x1f, 0x44, 0x00, 0x00
        .align 3, 0x90
        ret
# 6 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        # nopw 0(%[re]ax,%[re]ax,1)
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00
        .align 3, 0x90
        ret
# 7 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nopl 0L(%[re]ax)
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00
        .align 3, 0x90
        ret
# 8 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        # nopl 0L(%[re]ax,%[re]ax,1)
        # 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 3, 0x90
        ret
# 9 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        # nopw 0L(%[re]ax,%[re]ax,1)
        # 0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 10 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        # nopw %cs:0L(%[re]ax,%[re]ax,1)
        # 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 11 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        # nopw %cs:0L(%[re]ax,%[re]ax,1)
        # 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 12 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        # nopw 0(%[re]ax,%[re]ax,1)
        # nopw 0(%[re]ax,%[re]ax,1)
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00,
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00
        .align 4, 0x90
        ret
# 13 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        # nopw 0(%[re]ax,%[re]ax,1)
        # nopl 0L(%[re]ax)
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00,
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 14 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        # nopl 0L(%[re]ax)
        # nopl 0L(%[re]ax)
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00,
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 15 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nopl 0L(%[re]ax)
        # nopl 0L(%[re]ax,%[re]ax,1)
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00,
        # 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret

        # Only the .text sections gets optimal nops.
	.section	__TEXT,__const
f0:
        .byte 0
	.align	4, 0x90
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
// CHECK:     Size: 0x151
// CHECK:     Offset: 340
// CHECK:     Alignment: 4
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x800004)
// CHECK:       PureInstructions (0x800000)
// CHECK:       SomeInstructions (0x4)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:       0000: C390C300 00000000 00000000 00000000  |................|
// CHECK:       0010: C3C36690 C3000000 00000000 00000000  |..f.............|
// CHECK:       0020: C30F1F00 C3000000 00000000 00000000  |................|
// CHECK:       0030: C3C3C3C3 0F1F4000 C3000000 00000000  |......@.........|
// CHECK:       0040: C3C3C30F 1F440000 C3000000 00000000  |.....D..........|
// CHECK:       0050: C3C3660F 1F440000 C3000000 00000000  |..f..D..........|
// CHECK:       0060: C30F1F80 00000000 C3000000 00000000  |................|
// CHECK:       0070: C3C3C3C3 C3C3C3C3 C3000000 00000000  |................|
// CHECK:       0080: C3C3C3C3 C3C3C366 0F1F8400 00000000  |.......f........|
// CHECK:       0090: C3000000 00000000 00000000 00000000  |................|
// CHECK:       00A0: C3C3C3C3 C3C3C366 0F1F8400 00000000  |.......f........|
// CHECK:       00B0: C3000000 00000000 00000000 00000000  |................|
// CHECK:       00C0: C3C3C3C3 C3662E0F 1F840000 00000090  |.....f..........|
// CHECK:       00D0: C3000000 00000000 00000000 00000000  |................|
// CHECK:       00E0: C3C3C3C3 662E0F1F 84000000 00006690  |....f.........f.|
// CHECK:       00F0: C3000000 00000000 00000000 00000000  |................|
// CHECK:       0100: C3C3C366 2E0F1F84 00000000 000F1F00  |...f............|
// CHECK:       0110: C3000000 00000000 00000000 00000000  |................|
// CHECK:       0120: C3C3662E 0F1F8400 00000000 0F1F4000  |..f...........@.|
// CHECK:       0130: C3000000 00000000 00000000 00000000  |................|
// CHECK:       0140: C3662E0F 1F840000 0000000F 1F440000  |.f...........D..|
// CHECK:       0150: C3                                   |.|
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 1
// CHECK:     Name: __const (5F 5F 63 6F 6E 73 74 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x160
// CHECK:     Size: 0x14
// CHECK:     Offset: 692
// CHECK:     Alignment: 4
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 00909090 90909090 90909090 90909090  |................|
// CHECK:       0010: 00000000                             |....|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: f0 (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __const (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x160
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
// CHECK:   vmsize: 0x174
// CHECK:   fileoff: 340
// CHECK:   filesize: 372
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 2
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
