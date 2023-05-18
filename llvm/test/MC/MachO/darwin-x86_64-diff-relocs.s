// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj --file-headers -S --sd -r --symbols --macho-segment --macho-dysymtab --macho-indirect-symbols - | FileCheck %s

        .text

// FIXME: llvm-mc doesn't handle this in a way we can make compatible with 'as',
// currently, because of how we handle assembler variables.
//
// See <rdar://problem/7763719> improve handling of absolute symbols

// _baz = 4

_foo:
        xorl %eax,%eax
_g0:
        xorl %eax,%eax
L0:
        jmp 4
//        jmp _baz

// FIXME: Darwin 'as' for historical reasons widens this jump, but doesn't emit
// a relocation. It seems like 'as' widens any jump that is not to a temporary,
// which is inherited from the x86_32 behavior, even though x86_64 could do
// better.
//        jmp _g0

        jmp L0
        jmp _g1

// FIXME: Darwin 'as' gets this wrong as well, even though it could get it right
// given the other things we do on x86_64. It is using a short jump here. This
// is probably fallout of the hack that exists for x86_32.
//        jmp L1

// FIXME: We don't support this, and would currently get it wrong, it should be a jump to an absolute address.
//        jmp L0 - _g0

//        jmp _g1 - _g0
// FIXME: Darwin 'as' comes up with 'SIGNED' here instead of 'BRANCH'.
//        jmp _g1 - L1
// FIXME: Darwin 'as' gets this completely wrong. It ends up with a single
// branch relocation. Fallout from the other delta hack?
//        jmp L1 - _g0

        jmp _g2
        jmp L2
        jmp _g3
        jmp L3
// FIXME: Darwin 'as' gets this completely wrong. It ends up with a single
// branch relocation. Fallout from the other delta hack?
//        jmp L2 - _g3
//        jmp _g3 - _g2
// FIXME: Darwin 'as' comes up with 'SIGNED' here instead of 'BRANCH'.
//        jmp _g3 - L3
// FIXME: Darwin 'as' gets this completely wrong. It ends up with a single
// branch relocation. Fallout from the other delta hack?
//        jmp L3 - _g2

        movl %eax,4(%rip)
//        movl %eax,_baz(%rip)
        movl %eax,_g0(%rip)
        movl %eax,L0(%rip)
        movl %eax,_g1(%rip)
        movl %eax,L1(%rip)

// FIXME: Darwin 'as' gets most of these wrong, and there is an ambiguity in ATT
// syntax in what they should mean in the first place (absolute or
// rip-relative address).
//        movl %eax,L0 - _g0(%rip)
//        movl %eax,_g1 - _g0(%rip)
//        movl %eax,_g1 - L1(%rip)
//        movl %eax,L1 - _g0(%rip)

        movl %eax,_g2(%rip)
        movl %eax,L2(%rip)
        movl %eax,_g3(%rip)
        movl %eax,L3(%rip)

// FIXME: Darwin 'as' gets most of these wrong, and there is an ambiguity in ATT
// syntax in what they should mean in the first place (absolute or
// rip-relative address).
//        movl %eax,L2 - _g2(%rip)
//        movl %eax,_g3 - _g2(%rip)
//        movl %eax,_g3 - L3(%rip)
//        movl %eax,L3 - _g2(%rip)

_g1:
        xorl %eax,%eax
L1:
        xorl %eax,%eax

        .data
_g2:
        xorl %eax,%eax
L2:
        .quad 4
//        .quad _baz
        .quad _g2
        .quad L2
        .quad _g3
        .quad L3
        .quad L2 - _g2
        .quad _g3 - _g2
        .quad L3 - _g2
        .quad L3 - _g3

        .quad _g0
        .quad L0
        .quad _g1
        .quad L1
        .quad L0 - _g0
        .quad _g1 - _g0
        .quad L1 - _g0
        .quad L1 - _g1

_g3:
        xorl %eax,%eax
L3:
        xorl %eax,%eax

// FIXME: Unfortunately, we do not get these relocations in exactly the same
// order as Darwin 'as'. It turns out that 'as' *usually* ends up emitting
// them in reverse address order, but sometimes it allocates some
// additional relocations late so these end up precede the other entries. I
// haven't figured out the exact criteria for this yet.
 
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
// CHECK:     Size: 0x5E
// CHECK:     Offset: 384
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x270
// CHECK:     RelocationCount: 12
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x800004)
// CHECK:       PureInstructions (0x800000)
// CHECK:       SomeInstructions (0x4)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 31C031C0 E9040000 00EBF9E9 00000000  |1.1.............|
// CHECK:       0010: E9000000 00E90200 0000E900 000000E9  |................|
// CHECK:       0020: 02000000 89050400 00008905 D2FFFFFF  |................|
// CHECK:       0030: 8905CEFF FFFF8905 00000000 89050200  |................|
// CHECK:       0040: 00008905 00000000 89050200 00008905  |................|
// CHECK:       0050: 00000000 89050200 000031C0 31C0      |..........1.1.|
// CHECK:     )
// CHECK:   }
// CHECK:   Section {
// CHECK:     Index: 1
// CHECK:     Name: __data (5F 5F 64 61 74 61 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x5E
// CHECK:     Size: 0x8E
// CHECK:     Offset: 478
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x2D0
// CHECK:     RelocationCount: 16
// CHECK:     Type: Regular (0x0)
// CHECK:     Attributes [ (0x4)
// CHECK:       SomeInstructions (0x4)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     Reserved3: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 31C00400 00000000 00000000 00000000  |1...............|
// CHECK:       0010: 00000200 00000000 00000000 00000000  |................|
// CHECK:       0020: 00000200 00000000 00000200 00000000  |................|
// CHECK:       0030: 00000000 00000000 00000200 00000000  |................|
// CHECK:       0040: 00000200 00000000 00000000 00000000  |................|
// CHECK:       0050: 00000200 00000000 00000000 00000000  |................|
// CHECK:       0060: 00000200 00000000 00000200 00000000  |................|
// CHECK:       0070: 00000000 00000000 00000200 00000000  |................|
// CHECK:       0080: 00000200 00000000 000031C0 31C0      |..........1.1.|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK:   Section __text {
// CHECK:     0x56 1 2 1 X86_64_RELOC_SIGNED 0 _g3
// CHECK:     0x50 1 2 1 X86_64_RELOC_SIGNED 0 _g3
// CHECK:     0x4A 1 2 1 X86_64_RELOC_SIGNED 0 _g2
// CHECK:     0x44 1 2 1 X86_64_RELOC_SIGNED 0 _g2
// CHECK:     0x3E 1 2 1 X86_64_RELOC_SIGNED 0 _g1
// CHECK:     0x38 1 2 1 X86_64_RELOC_SIGNED 0 _g1
// CHECK:     0x20 1 2 1 X86_64_RELOC_BRANCH 0 _g3
// CHECK:     0x1B 1 2 1 X86_64_RELOC_BRANCH 0 _g3
// CHECK:     0x16 1 2 1 X86_64_RELOC_BRANCH 0 _g2
// CHECK:     0x11 1 2 1 X86_64_RELOC_BRANCH 0 _g2
// CHECK:     0xC 1 2 1 X86_64_RELOC_BRANCH 0 _g1
// CHECK:     0x5 1 2 1 X86_64_RELOC_BRANCH 0 _foo
// CHECK:   }
// CHECK:   Section __data {
// CHECK:     0x7A 0 3 1 X86_64_RELOC_SUBTRACTOR 0 _g0
// CHECK:     0x7A 0 3 1 X86_64_RELOC_UNSIGNED 0 _g1
// CHECK:     0x72 0 3 1 X86_64_RELOC_SUBTRACTOR 0 _g0
// CHECK:     0x72 0 3 1 X86_64_RELOC_UNSIGNED 0 _g1
// CHECK:     0x62 0 3 1 X86_64_RELOC_UNSIGNED 0 _g1
// CHECK:     0x5A 0 3 1 X86_64_RELOC_UNSIGNED 0 _g1
// CHECK:     0x52 0 3 1 X86_64_RELOC_UNSIGNED 0 _g0
// CHECK:     0x4A 0 3 1 X86_64_RELOC_UNSIGNED 0 _g0
// CHECK:     0x3A 0 3 1 X86_64_RELOC_SUBTRACTOR 0 _g2
// CHECK:     0x3A 0 3 1 X86_64_RELOC_UNSIGNED 0 _g3
// CHECK:     0x32 0 3 1 X86_64_RELOC_SUBTRACTOR 0 _g2
// CHECK:     0x32 0 3 1 X86_64_RELOC_UNSIGNED 0 _g3
// CHECK:     0x22 0 3 1 X86_64_RELOC_UNSIGNED 0 _g3
// CHECK:     0x1A 0 3 1 X86_64_RELOC_UNSIGNED 0 _g3
// CHECK:     0x12 0 3 1 X86_64_RELOC_UNSIGNED 0 _g2
// CHECK:     0xA 0 3 1 X86_64_RELOC_UNSIGNED 0 _g2
// CHECK:   }
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: _foo (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _g0 (18)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x2
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _g1 (14)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x5A
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _g2 (10)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x5E
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _g3 (6)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __data (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0xE8
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
// CHECK:   vmsize: 0xEC
// CHECK:   fileoff: 384
// CHECK:   filesize: 236
// CHECK:   maxprot: rwx
// CHECK:   initprot: rwx
// CHECK:   nsects: 2
// CHECK:   flags: 0x0
// CHECK: }
// CHECK: Dysymtab {
// CHECK:   ilocalsym: 0
// CHECK:   nlocalsym: 5
// CHECK:   iextdefsym: 5
// CHECK:   nextdefsym: 0
// CHECK:   iundefsym: 5
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
