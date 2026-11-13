// RUN: llvm-mc -triple aarch64 %s | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -filetype=obj -triple aarch64-linux-android %s -o - | llvm-readobj -S --sr --sd - | FileCheck %s
// RUN: not llvm-mc -triple=aarch64 -o - -defsym=ERR=1 %s 2>&1 | FileCheck %s --check-prefix=ERR

// ASM:      .cfi_lsda 3, bar
// ASM-NEXT: nop
// ASM:      .cfi_personality 0, foo
// ASM-NEXT: .cfi_lsda 3, bar

f1:
        .cfi_startproc
        .cfi_b_key_frame
        .cfi_lsda 0x3, bar
        nop
        .cfi_endproc

f2:
        .cfi_startproc
        .cfi_personality 0x00, foo
        .cfi_lsda 0x3, bar
        nop
        .cfi_endproc

f3:
        .cfi_startproc
        .cfi_lsda 0x3, bar
        .cfi_mte_tagged_frame
        nop
        .cfi_endproc

f4:
        .cfi_startproc
        .cfi_personality 0x00, foo
        .cfi_lsda 0x2, bar
        nop
        .cfi_endproc

f5:
        .cfi_startproc
        .cfi_personality 0x02, foo
        nop
        .cfi_endproc

f6:
        .cfi_startproc
        .cfi_personality 0x03, foo
        nop
        .cfi_endproc

f7:
        .cfi_startproc
        .cfi_personality 0x04, foo
        nop
        .cfi_endproc

f8:
        .cfi_startproc
        .cfi_personality 0x0a, foo
        nop
        .cfi_endproc

f9:
        .cfi_startproc
        .cfi_personality 0x0b, foo
        nop
        .cfi_endproc

f10:
        .cfi_startproc
        .cfi_personality 0x0c, foo
        nop
        .cfi_endproc

f11:
        .cfi_startproc
        .cfi_personality 0x08, foo
        nop
        .cfi_endproc

f12:
        .cfi_startproc
        .cfi_personality 0x10, foo
        nop
        .cfi_endproc

f13:
        .cfi_startproc
        .cfi_personality 0x12, foo
        nop
        .cfi_endproc

f14:
        .cfi_startproc
        .cfi_personality 0x13, foo
        nop
        .cfi_endproc

f15:
        .cfi_startproc
        .cfi_personality 0x14, foo
        nop
        .cfi_endproc

f16:
        .cfi_startproc
        .cfi_personality 0x1a, foo
        nop
        .cfi_endproc

f17:
        .cfi_startproc
        .cfi_personality 0x1b, foo
        nop
        .cfi_endproc

f18:
        .cfi_startproc
        .cfi_personality 0x1c, foo
        nop
        .cfi_endproc

f19:
        .cfi_startproc
        .cfi_personality 0x18, foo
        nop
        .cfi_endproc

f20:
        .cfi_startproc
        .cfi_personality 0x80, foo
        nop
        .cfi_endproc

f21:
        .cfi_startproc
        .cfi_personality 0x82, foo
        nop
        .cfi_endproc

f22:
        .cfi_startproc
        .cfi_personality 0x83, foo
        nop
        .cfi_endproc

f23:
        .cfi_startproc
        .cfi_personality 0x84, foo
        nop
        .cfi_endproc

f24:
        .cfi_startproc
        .cfi_personality 0x8a, foo
        nop
        .cfi_endproc

f25:
        .cfi_startproc
        .cfi_personality 0x8b, foo
        nop
        .cfi_endproc

f26:
        .cfi_startproc
        .cfi_personality 0x8c, foo
        nop
        .cfi_endproc

f27:
        .cfi_startproc
        .cfi_personality 0x88, foo
        nop
        .cfi_endproc

f28:
        .cfi_startproc
        .cfi_personality 0x90, foo
        nop
        .cfi_endproc

f29:
        .cfi_startproc
        .cfi_personality 0x92, foo
        nop
        .cfi_endproc

f30:
        .cfi_startproc
        .cfi_personality 0x93, foo
        nop
        .cfi_endproc

f31:
        .cfi_startproc
        .cfi_personality 0x94, foo
        nop
        .cfi_endproc

f32:
        .cfi_startproc
        .cfi_personality 0x9a, foo
        nop
        .cfi_endproc

f33:
        .cfi_startproc
        .cfi_personality 0x9b, foo
        nop
        .cfi_endproc

f34:
        .cfi_startproc
        .cfi_personality 0x9c, foo
        nop
        .cfi_endproc

f36:
        .cfi_startproc
        .cfi_personality 0x98, foo
        nop
        .cfi_endproc

f37:
        .cfi_startproc simple
        nop
        .cfi_endproc

f38:
        .cfi_startproc
        .cfi_non_hint_pauth_frame
        .cfi_lsda 0x3, bar
        nop
        .cfi_endproc

// CHECK:       Section {
// CHECK:         Name: .eh_frame
// CHECK-NEXT:    Type: SHT_PROGBITS (0x1)
// CHECK-NEXT:    Flags [ (0x2)
// CHECK-NEXT:      SHF_ALLOC (0x2)
// CHECK-NEXT:    ]
// CHECK-NEXT:    Address: 0x0
// CHECK-NEXT:    Offset: 0xD8
// CHECK-NEXT:    Size: 1808
// CHECK-NEXT:    Link: 0
// CHECK-NEXT:    Info: 0
// CHECK-NEXT:    AddressAlignment: 8
// CHECK-NEXT:    EntrySize: 0
// CHECK-NEXT:    Relocations [
// CHECK-NEXT:    ]
// CHECK-NEXT:    SectionData (
// CHECK-NEXT:      0000: 10000000 00000000 017A5200 017C1E01  |.........zR..|..|
// CHECK-NEXT:      0010: 1B000000 10000000 18000000 00000000  |................|
// CHECK-NEXT:      0020: 04000000 00000000 14000000 00000000  |................|
// CHECK-NEXT:      0030: 017A4C52 4E00017C 1E02031B 0C1F0000  |.zLRN..|........|
// CHECK-NEXT:      0040: 14000000 1C000000 00000000 04000000  |................|
// CHECK-NEXT:      0050: 04000000 00000000 14000000 00000000  |................|
// CHECK-NEXT:      0060: 017A4C52 4700017C 1E02031B 0C1F0000  |.zLRG..|........|
// CHECK-NEXT:      0070: 14000000 1C000000 00000000 04000000  |................|
// CHECK-NEXT:      0080: 04000000 00000000 14000000 00000000  |................|
// CHECK-NEXT:      0090: 017A4C52 4200017C 1E02031B 0C1F0000  |.zLRB..|........|
// CHECK-NEXT:      00A0: 14000000 1C000000 00000000 04000000  |................|
// CHECK-NEXT:      00B0: 04000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:      00C0: 017A504C 5200017C 1E0B0000 00000000  |.zPLR..|........|
// CHECK-NEXT:      00D0: 00000002 1B0C1F00 10000000 24000000  |............$...|
// CHECK-NEXT:      00E0: 00000000 04000000 02000000 1C000000  |................|
// CHECK-NEXT:      00F0: 00000000 017A504C 5200017C 1E0B0000  |.....zPLR..|....|
// CHECK-NEXT:      0100: 00000000 00000003 1B0C1F00 14000000  |................|
// CHECK-NEXT:      0110: 24000000 00000000 04000000 04000000  |$...............|
// CHECK-NEXT:      0120: 00000000 14000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      0130: 00017C1E 04020000 1B0C1F00 10000000  |..|.............|
// CHECK-NEXT:      0140: 1C000000 00000000 04000000 00000000  |................|
// CHECK-NEXT:      0150: 18000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      0160: 06030000 00001B0C 1F000000 10000000  |................|
// CHECK-NEXT:      0170: 20000000 00000000 04000000 00000000  | ...............|
// CHECK-NEXT:      0180: 1C000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      0190: 0A040000 00000000 00001B0C 1F000000  |................|
// CHECK-NEXT:      01A0: 10000000 24000000 00000000 04000000  |....$...........|
// CHECK-NEXT:      01B0: 00000000 1C000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      01C0: 00017C1E 0A080000 00000000 00001B0C  |..|.............|
// CHECK-NEXT:      01D0: 1F000000 10000000 24000000 00000000  |........$.......|
// CHECK-NEXT:      01E0: 04000000 00000000 14000000 00000000  |................|
// CHECK-NEXT:      01F0: 017A5052 00017C1E 040A0000 1B0C1F00  |.zPR..|.........|
// CHECK-NEXT:      0200: 10000000 1C000000 00000000 04000000  |................|
// CHECK-NEXT:      0210: 00000000 18000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      0220: 00017C1E 060B0000 00001B0C 1F000000  |..|.............|
// CHECK-NEXT:      0230: 10000000 20000000 00000000 04000000  |.... ...........|
// CHECK-NEXT:      0240: 00000000 1C000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      0250: 00017C1E 0A0C0000 00000000 00001B0C  |..|.............|
// CHECK-NEXT:      0260: 1F000000 10000000 24000000 00000000  |........$.......|
// CHECK-NEXT:      0270: 04000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:      0280: 017A5052 00017C1E 0A100000 00000000  |.zPR..|.........|
// CHECK-NEXT:      0290: 00001B0C 1F000000 10000000 24000000  |............$...|
// CHECK-NEXT:      02A0: 00000000 04000000 00000000 14000000  |................|
// CHECK-NEXT:      02B0: 00000000 017A5052 00017C1E 04120000  |.....zPR..|.....|
// CHECK-NEXT:      02C0: 1B0C1F00 10000000 1C000000 00000000  |................|
// CHECK-NEXT:      02D0: 04000000 00000000 18000000 00000000  |................|
// CHECK-NEXT:      02E0: 017A5052 00017C1E 06130000 00001B0C  |.zPR..|.........|
// CHECK-NEXT:      02F0: 1F000000 10000000 20000000 00000000  |........ .......|
// CHECK-NEXT:      0300: 04000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:      0310: 017A5052 00017C1E 0A140000 00000000  |.zPR..|.........|
// CHECK-NEXT:      0320: 00001B0C 1F000000 10000000 24000000  |............$...|
// CHECK-NEXT:      0330: 00000000 04000000 00000000 1C000000  |................|
// CHECK-NEXT:      0340: 00000000 017A5052 00017C1E 0A180000  |.....zPR..|.....|
// CHECK-NEXT:      0350: 00000000 00001B0C 1F000000 10000000  |................|
// CHECK-NEXT:      0360: 24000000 00000000 04000000 00000000  |$...............|
// CHECK-NEXT:      0370: 14000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      0380: 041A0000 1B0C1F00 10000000 1C000000  |................|
// CHECK-NEXT:      0390: 00000000 04000000 00000000 18000000  |................|
// CHECK-NEXT:      03A0: 00000000 017A5052 00017C1E 061B0000  |.....zPR..|.....|
// CHECK-NEXT:      03B0: 00001B0C 1F000000 10000000 20000000  |............ ...|
// CHECK-NEXT:      03C0: 00000000 04000000 00000000 1C000000  |................|
// CHECK-NEXT:      03D0: 00000000 017A5052 00017C1E 0A1C0000  |.....zPR..|.....|
// CHECK-NEXT:      03E0: 00000000 00001B0C 1F000000 10000000  |................|
// CHECK-NEXT:      03F0: 24000000 00000000 04000000 00000000  |$...............|
// CHECK-NEXT:      0400: 1C000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      0410: 0A800000 00000000 00001B0C 1F000000  |................|
// CHECK-NEXT:      0420: 10000000 24000000 00000000 04000000  |....$...........|
// CHECK-NEXT:      0430: 00000000 14000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      0440: 00017C1E 04820000 1B0C1F00 10000000  |..|.............|
// CHECK-NEXT:      0450: 1C000000 00000000 04000000 00000000  |................|
// CHECK-NEXT:      0460: 18000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      0470: 06830000 00001B0C 1F000000 10000000  |................|
// CHECK-NEXT:      0480: 20000000 00000000 04000000 00000000  | ...............|
// CHECK-NEXT:      0490: 1C000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      04A0: 0A840000 00000000 00001B0C 1F000000  |................|
// CHECK-NEXT:      04B0: 10000000 24000000 00000000 04000000  |....$...........|
// CHECK-NEXT:      04C0: 00000000 1C000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      04D0: 00017C1E 0A880000 00000000 00001B0C  |..|.............|
// CHECK-NEXT:      04E0: 1F000000 10000000 24000000 00000000  |........$.......|
// CHECK-NEXT:      04F0: 04000000 00000000 14000000 00000000  |................|
// CHECK-NEXT:      0500: 017A5052 00017C1E 048A0000 1B0C1F00  |.zPR..|.........|
// CHECK-NEXT:      0510: 10000000 1C000000 00000000 04000000  |................|
// CHECK-NEXT:      0520: 00000000 18000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      0530: 00017C1E 068B0000 00001B0C 1F000000  |..|.............|
// CHECK-NEXT:      0540: 10000000 20000000 00000000 04000000  |.... ...........|
// CHECK-NEXT:      0550: 00000000 1C000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      0560: 00017C1E 0A8C0000 00000000 00001B0C  |..|.............|
// CHECK-NEXT:      0570: 1F000000 10000000 24000000 00000000  |........$.......|
// CHECK-NEXT:      0580: 04000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:      0590: 017A5052 00017C1E 0A900000 00000000  |.zPR..|.........|
// CHECK-NEXT:      05A0: 00001B0C 1F000000 10000000 24000000  |............$...|
// CHECK-NEXT:      05B0: 00000000 04000000 00000000 14000000  |................|
// CHECK-NEXT:      05C0: 00000000 017A5052 00017C1E 04920000  |.....zPR..|.....|
// CHECK-NEXT:      05D0: 1B0C1F00 10000000 1C000000 00000000  |................|
// CHECK-NEXT:      05E0: 04000000 00000000 18000000 00000000  |................|
// CHECK-NEXT:      05F0: 017A5052 00017C1E 06930000 00001B0C  |.zPR..|.........|
// CHECK-NEXT:      0600: 1F000000 10000000 20000000 00000000  |........ .......|
// CHECK-NEXT:      0610: 04000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:      0620: 017A5052 00017C1E 0A940000 00000000  |.zPR..|.........|
// CHECK-NEXT:      0630: 00001B0C 1F000000 10000000 24000000  |............$...|
// CHECK-NEXT:      0640: 00000000 04000000 00000000 1C000000  |................|
// CHECK-NEXT:      0650: 00000000 017A5052 00017C1E 0A980000  |.....zPR..|.....|
// CHECK-NEXT:      0660: 00000000 00001B0C 1F000000 10000000  |................|
// CHECK-NEXT:      0670: 24000000 00000000 04000000 00000000  |$...............|
// CHECK-NEXT:      0680: 14000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      0690: 049A0000 1B0C1F00 10000000 1C000000  |................|
// CHECK-NEXT:      06A0: 00000000 04000000 00000000 18000000  |................|
// CHECK-NEXT:      06B0: 00000000 017A5052 00017C1E 069B0000  |.....zPR..|.....|
// CHECK-NEXT:      06C0: 00001B0C 1F000000 10000000 20000000  |............ ...|
// CHECK-NEXT:      06D0: 00000000 04000000 00000000 1C000000  |................|
// CHECK-NEXT:      06E0: 00000000 017A5052 00017C1E 0A9C0000  |.....zPR..|.....|
// CHECK-NEXT:      06F0: 00000000 00001B0C 1F000000 10000000  |................|
// CHECK-NEXT:      0700: 24000000 00000000 04000000 00000000  |$...............|
// CHECK-NEXT:    )
// CHECK-NEXT:  }
// CHECK:       Section {
// CHECK:         Name: .rela.eh_frame
// CHECK-NEXT:    Type: SHT_RELA (0x4)
// CHECK-NEXT:    Flags [ (0x40)
// CHECK-NEXT:      SHF_INFO_LINK (0x40)
// CHECK-NEXT:    ]
// CHECK-NEXT:    Address: 0x0
// CHECK-NEXT:    Offset: 0xBF0
// CHECK-NEXT:    Size: 1800
// CHECK-NEXT:    Link: 5
// CHECK-NEXT:    Info: 3
// CHECK-NEXT:    AddressAlignment: 8
// CHECK-NEXT:    EntrySize: 24
// CHECK-NEXT:    Relocations [
// CHECK-NEXT:      0x1C R_AARCH64_PREL32 .text 0x8C
// CHECK-NEXT:      0x48 R_AARCH64_PREL32 .text 0x90
// CHECK-NEXT:      0x51 R_AARCH64_ABS32 bar 0x0
// CHECK-NEXT:      0x78 R_AARCH64_PREL32 .text 0x8
// CHECK-NEXT:      0x81 R_AARCH64_ABS32 bar 0x0
// CHECK-NEXT:      0xA8 R_AARCH64_PREL32 .text 0x0
// CHECK-NEXT:      0xB1 R_AARCH64_ABS32 bar 0x0
// CHECK-NEXT:      0xCB R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0xE0 R_AARCH64_PREL32 .text 0xC
// CHECK-NEXT:      0xE9 R_AARCH64_ABS16 bar 0x0
// CHECK-NEXT:      0xFF R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x114 R_AARCH64_PREL32 .text 0x4
// CHECK-NEXT:      0x11D R_AARCH64_ABS32 bar 0x0
// CHECK-NEXT:      0x136 R_AARCH64_ABS16 foo 0x0
// CHECK-NEXT:      0x144 R_AARCH64_PREL32 .text 0x10
// CHECK-NEXT:      0x162 R_AARCH64_ABS32 foo 0x0
// CHECK-NEXT:      0x174 R_AARCH64_PREL32 .text 0x14
// CHECK-NEXT:      0x192 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x1A8 R_AARCH64_PREL32 .text 0x18
// CHECK-NEXT:      0x1C6 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x1DC R_AARCH64_PREL32 .text 0x28
// CHECK-NEXT:      0x1FA R_AARCH64_ABS16 foo 0x0
// CHECK-NEXT:      0x208 R_AARCH64_PREL32 .text 0x1C
// CHECK-NEXT:      0x226 R_AARCH64_ABS32 foo 0x0
// CHECK-NEXT:      0x238 R_AARCH64_PREL32 .text 0x20
// CHECK-NEXT:      0x256 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x26C R_AARCH64_PREL32 .text 0x24
// CHECK-NEXT:      0x28A R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x2A0 R_AARCH64_PREL32 .text 0x2C
// CHECK-NEXT:      0x2BE R_AARCH64_PREL16 foo 0x0
// CHECK-NEXT:      0x2CC R_AARCH64_PREL32 .text 0x30
// CHECK-NEXT:      0x2EA R_AARCH64_PREL32 foo 0x0
// CHECK-NEXT:      0x2FC R_AARCH64_PREL32 .text 0x34
// CHECK-NEXT:      0x31A R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x330 R_AARCH64_PREL32 .text 0x38
// CHECK-NEXT:      0x34E R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x364 R_AARCH64_PREL32 .text 0x48
// CHECK-NEXT:      0x382 R_AARCH64_PREL16 foo 0x0
// CHECK-NEXT:      0x390 R_AARCH64_PREL32 .text 0x3C
// CHECK-NEXT:      0x3AE R_AARCH64_PREL32 foo 0x0
// CHECK-NEXT:      0x3C0 R_AARCH64_PREL32 .text 0x40
// CHECK-NEXT:      0x3DE R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x3F4 R_AARCH64_PREL32 .text 0x44
// CHECK-NEXT:      0x412 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x428 R_AARCH64_PREL32 .text 0x4C
// CHECK-NEXT:      0x446 R_AARCH64_ABS16 foo 0x0
// CHECK-NEXT:      0x454 R_AARCH64_PREL32 .text 0x50
// CHECK-NEXT:      0x472 R_AARCH64_ABS32 foo 0x0
// CHECK-NEXT:      0x484 R_AARCH64_PREL32 .text 0x54
// CHECK-NEXT:      0x4A2 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x4B8 R_AARCH64_PREL32 .text 0x58
// CHECK-NEXT:      0x4D6 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x4EC R_AARCH64_PREL32 .text 0x68
// CHECK-NEXT:      0x50A R_AARCH64_ABS16 foo 0x0
// CHECK-NEXT:      0x518 R_AARCH64_PREL32 .text 0x5C
// CHECK-NEXT:      0x536 R_AARCH64_ABS32 foo 0x0
// CHECK-NEXT:      0x548 R_AARCH64_PREL32 .text 0x60
// CHECK-NEXT:      0x566 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x57C R_AARCH64_PREL32 .text 0x64
// CHECK-NEXT:      0x59A R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x5B0 R_AARCH64_PREL32 .text 0x6C
// CHECK-NEXT:      0x5CE R_AARCH64_PREL16 foo 0x0
// CHECK-NEXT:      0x5DC R_AARCH64_PREL32 .text 0x70
// CHECK-NEXT:      0x5FA R_AARCH64_PREL32 foo 0x0
// CHECK-NEXT:      0x60C R_AARCH64_PREL32 .text 0x74
// CHECK-NEXT:      0x62A R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x640 R_AARCH64_PREL32 .text 0x78
// CHECK-NEXT:      0x65E R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x674 R_AARCH64_PREL32 .text 0x88
// CHECK-NEXT:      0x692 R_AARCH64_PREL16 foo 0x0
// CHECK-NEXT:      0x6A0 R_AARCH64_PREL32 .text 0x7C
// CHECK-NEXT:      0x6BE R_AARCH64_PREL32 foo 0x0
// CHECK-NEXT:      0x6D0 R_AARCH64_PREL32 .text 0x80
// CHECK-NEXT:      0x6EE R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x704 R_AARCH64_PREL32 .text 0x84
// CHECK-NEXT:    ]
// CHECK-NEXT:    SectionData (
// CHECK-NEXT:      0000: 1C000000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0010: 8C000000 00000000 48000000 00000000  |........H.......|
// CHECK-NEXT:      0020: 05010000 01000000 90000000 00000000  |................|
// CHECK-NEXT:      0030: 51000000 00000000 02010000 29000000  |Q...........)...|
// CHECK-NEXT:      0040: 00000000 00000000 78000000 00000000  |........x.......|
// CHECK-NEXT:      0050: 05010000 01000000 08000000 00000000  |................|
// CHECK-NEXT:      0060: 81000000 00000000 02010000 29000000  |............)...|
// CHECK-NEXT:      0070: 00000000 00000000 A8000000 00000000  |................|
// CHECK-NEXT:      0080: 05010000 01000000 00000000 00000000  |................|
// CHECK-NEXT:      0090: B1000000 00000000 02010000 29000000  |............)...|
// CHECK-NEXT:      00A0: 00000000 00000000 CB000000 00000000  |................|
// CHECK-NEXT:      00B0: 01010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      00C0: E0000000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      00D0: 0C000000 00000000 E9000000 00000000  |................|
// CHECK-NEXT:      00E0: 03010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      00F0: FF000000 00000000 01010000 2A000000  |............*...|
// CHECK-NEXT:      0100: 00000000 00000000 14010000 00000000  |................|
// CHECK-NEXT:      0110: 05010000 01000000 04000000 00000000  |................|
// CHECK-NEXT:      0120: 1D010000 00000000 02010000 29000000  |............)...|
// CHECK-NEXT:      0130: 00000000 00000000 36010000 00000000  |........6.......|
// CHECK-NEXT:      0140: 03010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0150: 44010000 00000000 05010000 01000000  |D...............|
// CHECK-NEXT:      0160: 10000000 00000000 62010000 00000000  |........b.......|
// CHECK-NEXT:      0170: 02010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0180: 74010000 00000000 05010000 01000000  |t...............|
// CHECK-NEXT:      0190: 14000000 00000000 92010000 00000000  |................|
// CHECK-NEXT:      01A0: 01010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      01B0: A8010000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      01C0: 18000000 00000000 C6010000 00000000  |................|
// CHECK-NEXT:      01D0: 01010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      01E0: DC010000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      01F0: 28000000 00000000 FA010000 00000000  |(...............|
// CHECK-NEXT:      0200: 03010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0210: 08020000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0220: 1C000000 00000000 26020000 00000000  |........&.......|
// CHECK-NEXT:      0230: 02010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0240: 38020000 00000000 05010000 01000000  |8...............|
// CHECK-NEXT:      0250: 20000000 00000000 56020000 00000000  | .......V.......|
// CHECK-NEXT:      0260: 01010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0270: 6C020000 00000000 05010000 01000000  |l...............|
// CHECK-NEXT:      0280: 24000000 00000000 8A020000 00000000  |$...............|
// CHECK-NEXT:      0290: 04010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      02A0: A0020000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      02B0: 2C000000 00000000 BE020000 00000000  |,...............|
// CHECK-NEXT:      02C0: 06010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      02D0: CC020000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      02E0: 30000000 00000000 EA020000 00000000  |0...............|
// CHECK-NEXT:      02F0: 05010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0300: FC020000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0310: 34000000 00000000 1A030000 00000000  |4...............|
// CHECK-NEXT:      0320: 04010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0330: 30030000 00000000 05010000 01000000  |0...............|
// CHECK-NEXT:      0340: 38000000 00000000 4E030000 00000000  |8.......N.......|
// CHECK-NEXT:      0350: 04010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0360: 64030000 00000000 05010000 01000000  |d...............|
// CHECK-NEXT:      0370: 48000000 00000000 82030000 00000000  |H...............|
// CHECK-NEXT:      0380: 06010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0390: 90030000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      03A0: 3C000000 00000000 AE030000 00000000  |<...............|
// CHECK-NEXT:      03B0: 05010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      03C0: C0030000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      03D0: 40000000 00000000 DE030000 00000000  |@...............|
// CHECK-NEXT:      03E0: 04010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      03F0: F4030000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0400: 44000000 00000000 12040000 00000000  |D...............|
// CHECK-NEXT:      0410: 01010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0420: 28040000 00000000 05010000 01000000  |(...............|
// CHECK-NEXT:      0430: 4C000000 00000000 46040000 00000000  |L.......F.......|
// CHECK-NEXT:      0440: 03010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0450: 54040000 00000000 05010000 01000000  |T...............|
// CHECK-NEXT:      0460: 50000000 00000000 72040000 00000000  |P.......r.......|
// CHECK-NEXT:      0470: 02010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0480: 84040000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0490: 54000000 00000000 A2040000 00000000  |T...............|
// CHECK-NEXT:      04A0: 01010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      04B0: B8040000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      04C0: 58000000 00000000 D6040000 00000000  |X...............|
// CHECK-NEXT:      04D0: 01010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      04E0: EC040000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      04F0: 68000000 00000000 0A050000 00000000  |h...............|
// CHECK-NEXT:      0500: 03010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0510: 18050000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0520: 5C000000 00000000 36050000 00000000  |\.......6.......|
// CHECK-NEXT:      0530: 02010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0540: 48050000 00000000 05010000 01000000  |H...............|
// CHECK-NEXT:      0550: 60000000 00000000 66050000 00000000  |`.......f.......|
// CHECK-NEXT:      0560: 01010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0570: 7C050000 00000000 05010000 01000000  ||...............|
// CHECK-NEXT:      0580: 64000000 00000000 9A050000 00000000  |d...............|
// CHECK-NEXT:      0590: 04010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      05A0: B0050000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      05B0: 6C000000 00000000 CE050000 00000000  |l...............|
// CHECK-NEXT:      05C0: 06010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      05D0: DC050000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      05E0: 70000000 00000000 FA050000 00000000  |p...............|
// CHECK-NEXT:      05F0: 05010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0600: 0C060000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0610: 74000000 00000000 2A060000 00000000  |t.......*.......|
// CHECK-NEXT:      0620: 04010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0630: 40060000 00000000 05010000 01000000  |@...............|
// CHECK-NEXT:      0640: 78000000 00000000 5E060000 00000000  |x.......^.......|
// CHECK-NEXT:      0650: 04010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0660: 74060000 00000000 05010000 01000000  |t...............|
// CHECK-NEXT:      0670: 88000000 00000000 92060000 00000000  |................|
// CHECK-NEXT:      0680: 06010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      0690: A0060000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      06A0: 7C000000 00000000 BE060000 00000000  ||...............|
// CHECK-NEXT:      06B0: 05010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      06C0: D0060000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      06D0: 80000000 00000000 EE060000 00000000  |................|
// CHECK-NEXT:      06E0: 04010000 2A000000 00000000 00000000  |....*...........|
// CHECK-NEXT:      06F0: 04070000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0700: 84000000 00000000                    |........|
// CHECK-NEXT:    )
// CHECK-NEXT:  }

.ifdef ERR
// ERR: [[#@LINE+1]]:15: error: expected .eh_frame, .debug_frame, or .sframe
.cfi_sections $
// ERR: [[#@LINE+1]]:28: error: expected comma
.cfi_sections .debug_frame $
// ERR: [[#@LINE+1]]:39: error: expected comma
.cfi_sections .debug_frame, .eh_frame $
// ERR: [[#@LINE+1]]:48: error: expected comma
.cfi_sections .debug_frame, .eh_frame, .sframe $

// ERR: [[#@LINE+1]]:16: error: unexpected token
.cfi_startproc $
// ERR: [[#@LINE+1]]:23: error: expected newline
.cfi_startproc simple $

// ERR: [[#@LINE+1]]:14: error: expected newline
.cfi_endproc $
.endif
