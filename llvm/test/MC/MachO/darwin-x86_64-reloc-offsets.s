// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj --file-headers -S --sd -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols - | FileCheck %s

        .data

        .org 0x10
L0:
        .long 0
        .long 0
        .long 0
        .long 0

_d:
        .long 0
L1:
        .long 0

        .text

// These generate normal x86_64 (external) relocations. They could all use
// SIGNED, but don't for pedantic compatibility with Darwin 'as'.

        // SIGNED1
 	movb  $0x12, _d(%rip)

        // SIGNED
 	movb  $0x12, _d + 1(%rip)

        // SIGNED4
 	movl  $0x12345678, _d(%rip)

        // SIGNED
 	movl  $0x12345678, _d + 1(%rip)

        // SIGNED2
 	movl  $0x12345678, _d + 2(%rip)

        // SIGNED1
 	movl  $0x12345678, _d + 3(%rip)

        // SIGNED
 	movl  $0x12345678, _d + 4(%rip)

	movb  %al, _d(%rip)
 	movb  %al, _d + 1(%rip)
 	movl  %eax, _d(%rip)
 	movl  %eax, _d + 1(%rip)
 	movl  %eax, _d + 2(%rip)
 	movl  %eax, _d + 3(%rip)
 	movl  %eax, _d + 4(%rip)

// These have to use local relocations. Since that uses an offset into the
// section in x86_64 (as opposed to a scattered relocation), and since the
// linker can only decode this to an atom + offset by scanning the section,
// it is not possible to correctly encode these without SIGNED<N>. This is
// ultimately due to a design flaw in the x86_64 relocation format, it is
// not possible to encode an address (L<foo> + <constant>) which is outside the
// atom containing L<foo>.

        // SIGNED1
 	movb  $0x12, L0(%rip)

        // SIGNED
 	movb  $0x12, L0 + 1(%rip)

        // SIGNED4
 	movl  $0x12345678, L0(%rip)

        // SIGNED
 	movl  $0x12345678, L0 + 1(%rip)

        // SIGNED2
 	movl  $0x12345678, L0 + 2(%rip)

        // SIGNED1
 	movl  $0x12345678, L0 + 3(%rip)

        // SIGNED
 	movl  $0x12345678, L0 + 4(%rip)

 	movb  %al, L0(%rip)
 	movb  %al, L0 + 1(%rip)
 	movl  %eax, L0(%rip)
 	movl  %eax, L0 + 1(%rip)
 	movl  %eax, L0 + 2(%rip)
 	movl  %eax, L0 + 3(%rip)
 	movl  %eax, L0 + 4(%rip)

        // SIGNED1
 	movb  $0x12, L1(%rip)

        // SIGNED
 	movb  $0x12, L1 + 1(%rip)

        // SIGNED4
 	movl  $0x12345678, L1(%rip)

        // SIGNED
 	movl  $0x12345678, L1 + 1(%rip)

        // SIGNED2
 	movl  $0x12345678, L1 + 2(%rip)

        // SIGNED1
 	movl  $0x12345678, L1 + 3(%rip)

        // SIGNED
 	movl  $0x12345678, L1 + 4(%rip)

 	movb  %al, L1(%rip)
 	movb  %al, L1 + 1(%rip)
 	movl  %eax, L1(%rip)
 	movl  %eax, L1 + 1(%rip)
 	movl  %eax, L1 + 2(%rip)
 	movl  %eax, L1 + 3(%rip)
 	movl  %eax, L1 + 4(%rip)

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
// CHECK:     Size: 0x13E
// CHECK:     Offset: 384
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x2E8
// CHECK:     RelocationCount: 42
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x800004)
// CHECK:       PureInstructions (0x800000)
// CHECK:       SomeInstructions (0x4)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:       0000: C605FFFF FFFF12C6 05000000 0012C705  |................|
// CHECK:       0010: FCFFFFFF 78563412 C705FDFF FFFF7856  |....xV4.......xV|
// CHECK:       0020: 3412C705 FEFFFFFF 78563412 C705FFFF  |4.......xV4.....|
// CHECK:       0030: FFFF7856 3412C705 00000000 78563412  |..xV4.......xV4.|
// CHECK:       0040: 88050000 00008805 01000000 89050000  |................|
// CHECK:       0050: 00008905 01000000 89050200 00008905  |................|
// CHECK:       0060: 03000000 89050400 0000C605 DD000000  |................|
// CHECK:       0070: 12C605D7 00000012 C705CC00 00007856  |..............xV|
// CHECK:       0080: 3412C705 C3000000 78563412 C705BA00  |4.......xV4.....|
// CHECK:       0090: 00007856 3412C705 B1000000 78563412  |..xV4.......xV4.|
// CHECK:       00A0: C705A800 00007856 34128805 9E000000  |......xV4.......|
// CHECK:       00B0: 88059900 00008905 92000000 89058D00  |................|
// CHECK:       00C0: 00008905 88000000 89058300 00008905  |................|
// CHECK:       00D0: 7E000000 C6050300 000012C6 05040000  |~...............|
// CHECK:       00E0: 0012C705 00000000 78563412 C7050100  |........xV4.....|
// CHECK:       00F0: 00007856 3412C705 02000000 78563412  |..xV4.......xV4.|
// CHECK:       0100: C7050300 00007856 3412C705 04000000  |......xV4.......|
// CHECK:       0110: 78563412 88050400 00008805 05000000  |xV4.............|
// CHECK:       0120: 89050400 00008905 05000000 89050600  |................|
// CHECK:       0130: 00008905 07000000 89050800 0000      |..............|
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 1
// CHECK:     Name: __data (5F 5F 64 61 74 61 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x13E
// CHECK:     Size: 0x28
// CHECK:     Offset: 702
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x0
// CHECK:     RelocationCount: 0
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x0)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 00000000 00000000 00000000 00000000  |................|
// CHECK:       0010: 00000000 00000000 00000000 00000000  |................|
// CHECK:       0020: 00000000 00000000                    |........|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK:   Section __text {
// CHECK:     0x13A 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x134 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x12E 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x128 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x122 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x11C 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x116 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x10C 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x102 1 2 1 X86_64_RELOC_SIGNED_1 0 _d
// CHECK:     0xF8 1 2 1 X86_64_RELOC_SIGNED_2 0 _d
// CHECK:     0xEE 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0xE4 1 2 1 X86_64_RELOC_SIGNED_4 0 _d
// CHECK:     0xDD 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0xD6 1 2 1 X86_64_RELOC_SIGNED_1 0 _d
// CHECK:     0xD0 1 2 0 X86_64_RELOC_SIGNED 0 __data
// CHECK:     0xCA 1 2 0 X86_64_RELOC_SIGNED 0 __data
// CHECK:     0xC4 1 2 0 X86_64_RELOC_SIGNED 0 __data
// CHECK:     0xBE 1 2 0 X86_64_RELOC_SIGNED 0 __data
// CHECK:     0xB8 1 2 0 X86_64_RELOC_SIGNED 0 __data
// CHECK:     0xB2 1 2 0 X86_64_RELOC_SIGNED 0 __data
// CHECK:     0xAC 1 2 0 X86_64_RELOC_SIGNED 0 __data
// CHECK:     0xA2 1 2 0 X86_64_RELOC_SIGNED 0 __data
// CHECK:     0x98 1 2 0 X86_64_RELOC_SIGNED_1 0 __data
// CHECK:     0x8E 1 2 0 X86_64_RELOC_SIGNED_2 0 __data
// CHECK:     0x84 1 2 0 X86_64_RELOC_SIGNED 0 __data
// CHECK:     0x7A 1 2 0 X86_64_RELOC_SIGNED_4 0 __data
// CHECK:     0x73 1 2 0 X86_64_RELOC_SIGNED 0 __data
// CHECK:     0x6C 1 2 0 X86_64_RELOC_SIGNED_1 0 __data
// CHECK:     0x66 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x60 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x5A 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x54 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x4E 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x48 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x42 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x38 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x2E 1 2 1 X86_64_RELOC_SIGNED_1 0 _d
// CHECK:     0x24 1 2 1 X86_64_RELOC_SIGNED_2 0 _d
// CHECK:     0x1A 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x10 1 2 1 X86_64_RELOC_SIGNED_4 0 _d
// CHECK:     0x9 1 2 1 X86_64_RELOC_SIGNED 0 _d
// CHECK:     0x2 1 2 1 X86_64_RELOC_SIGNED_1 0 _d
// CHECK:   }
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: _d (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x15E
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
// CHECK:   vmsize: 0x166
// CHECK:   fileoff: 384
// CHECK:   filesize: 358
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
