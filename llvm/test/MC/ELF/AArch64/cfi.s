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

// CHECK:       Section {
// CHECK:         Name: .eh_frame (20)
// CHECK-NEXT:    Type: SHT_PROGBITS (0x1)
// CHECK-NEXT:    Flags [ (0x2)
// CHECK-NEXT:      SHF_ALLOC (0x2)
// CHECK-NEXT:    ]
// CHECK-NEXT:    Address: 0x0
// CHECK-NEXT:    Offset: 0xD0
// CHECK-NEXT:    Size: 1760
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
// CHECK-NEXT:      0030: 017A4C52 00017C1E 02031B0C 1F000000  |.zLR..|.........|
// CHECK-NEXT:      0040: 14000000 1C000000 00000000 04000000  |................|
// CHECK-NEXT:      0050: 04000000 00000000 14000000 00000000  |................|
// CHECK-NEXT:      0060: 017A4C52 4200017C 1E02031B 0C1F0000  |.zLRB..|........|
// CHECK-NEXT:      0070: 14000000 1C000000 00000000 04000000  |................|
// CHECK-NEXT:      0080: 04000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:      0090: 017A504C 5200017C 1E0B0000 00000000  |.zPLR..|........|
// CHECK-NEXT:      00A0: 00000002 1B0C1F00 10000000 24000000  |............$...|
// CHECK-NEXT:      00B0: 00000000 04000000 02000000 1C000000  |................|
// CHECK-NEXT:      00C0: 00000000 017A504C 5200017C 1E0B0000  |.....zPLR..|....|
// CHECK-NEXT:      00D0: 00000000 00000003 1B0C1F00 14000000  |................|
// CHECK-NEXT:      00E0: 24000000 00000000 04000000 04000000  |$...............|
// CHECK-NEXT:      00F0: 00000000 14000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      0100: 00017C1E 04020000 1B0C1F00 10000000  |..|.............|
// CHECK-NEXT:      0110: 1C000000 00000000 04000000 00000000  |................|
// CHECK-NEXT:      0120: 18000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      0130: 06030000 00001B0C 1F000000 10000000  |................|
// CHECK-NEXT:      0140: 20000000 00000000 04000000 00000000  | ...............|
// CHECK-NEXT:      0150: 1C000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      0160: 0A040000 00000000 00001B0C 1F000000  |................|
// CHECK-NEXT:      0170: 10000000 24000000 00000000 04000000  |....$...........|
// CHECK-NEXT:      0180: 00000000 1C000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      0190: 00017C1E 0A080000 00000000 00001B0C  |..|.............|
// CHECK-NEXT:      01A0: 1F000000 10000000 24000000 00000000  |........$.......|
// CHECK-NEXT:      01B0: 04000000 00000000 14000000 00000000  |................|
// CHECK-NEXT:      01C0: 017A5052 00017C1E 040A0000 1B0C1F00  |.zPR..|.........|
// CHECK-NEXT:      01D0: 10000000 1C000000 00000000 04000000  |................|
// CHECK-NEXT:      01E0: 00000000 18000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      01F0: 00017C1E 060B0000 00001B0C 1F000000  |..|.............|
// CHECK-NEXT:      0200: 10000000 20000000 00000000 04000000  |.... ...........|
// CHECK-NEXT:      0210: 00000000 1C000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      0220: 00017C1E 0A0C0000 00000000 00001B0C  |..|.............|
// CHECK-NEXT:      0230: 1F000000 10000000 24000000 00000000  |........$.......|
// CHECK-NEXT:      0240: 04000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:      0250: 017A5052 00017C1E 0A100000 00000000  |.zPR..|.........|
// CHECK-NEXT:      0260: 00001B0C 1F000000 10000000 24000000  |............$...|
// CHECK-NEXT:      0270: 00000000 04000000 00000000 14000000  |................|
// CHECK-NEXT:      0280: 00000000 017A5052 00017C1E 04120000  |.....zPR..|.....|
// CHECK-NEXT:      0290: 1B0C1F00 10000000 1C000000 00000000  |................|
// CHECK-NEXT:      02A0: 04000000 00000000 18000000 00000000  |................|
// CHECK-NEXT:      02B0: 017A5052 00017C1E 06130000 00001B0C  |.zPR..|.........|
// CHECK-NEXT:      02C0: 1F000000 10000000 20000000 00000000  |........ .......|
// CHECK-NEXT:      02D0: 04000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:      02E0: 017A5052 00017C1E 0A140000 00000000  |.zPR..|.........|
// CHECK-NEXT:      02F0: 00001B0C 1F000000 10000000 24000000  |............$...|
// CHECK-NEXT:      0300: 00000000 04000000 00000000 1C000000  |................|
// CHECK-NEXT:      0310: 00000000 017A5052 00017C1E 0A180000  |.....zPR..|.....|
// CHECK-NEXT:      0320: 00000000 00001B0C 1F000000 10000000  |................|
// CHECK-NEXT:      0330: 24000000 00000000 04000000 00000000  |$...............|
// CHECK-NEXT:      0340: 14000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      0350: 041A0000 1B0C1F00 10000000 1C000000  |................|
// CHECK-NEXT:      0360: 00000000 04000000 00000000 18000000  |................|
// CHECK-NEXT:      0370: 00000000 017A5052 00017C1E 061B0000  |.....zPR..|.....|
// CHECK-NEXT:      0380: 00001B0C 1F000000 10000000 20000000  |............ ...|
// CHECK-NEXT:      0390: 00000000 04000000 00000000 1C000000  |................|
// CHECK-NEXT:      03A0: 00000000 017A5052 00017C1E 0A1C0000  |.....zPR..|.....|
// CHECK-NEXT:      03B0: 00000000 00001B0C 1F000000 10000000  |................|
// CHECK-NEXT:      03C0: 24000000 00000000 04000000 00000000  |$...............|
// CHECK-NEXT:      03D0: 1C000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      03E0: 0A800000 00000000 00001B0C 1F000000  |................|
// CHECK-NEXT:      03F0: 10000000 24000000 00000000 04000000  |....$...........|
// CHECK-NEXT:      0400: 00000000 14000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      0410: 00017C1E 04820000 1B0C1F00 10000000  |..|.............|
// CHECK-NEXT:      0420: 1C000000 00000000 04000000 00000000  |................|
// CHECK-NEXT:      0430: 18000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      0440: 06830000 00001B0C 1F000000 10000000  |................|
// CHECK-NEXT:      0450: 20000000 00000000 04000000 00000000  | ...............|
// CHECK-NEXT:      0460: 1C000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      0470: 0A840000 00000000 00001B0C 1F000000  |................|
// CHECK-NEXT:      0480: 10000000 24000000 00000000 04000000  |....$...........|
// CHECK-NEXT:      0490: 00000000 1C000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      04A0: 00017C1E 0A880000 00000000 00001B0C  |..|.............|
// CHECK-NEXT:      04B0: 1F000000 10000000 24000000 00000000  |........$.......|
// CHECK-NEXT:      04C0: 04000000 00000000 14000000 00000000  |................|
// CHECK-NEXT:      04D0: 017A5052 00017C1E 048A0000 1B0C1F00  |.zPR..|.........|
// CHECK-NEXT:      04E0: 10000000 1C000000 00000000 04000000  |................|
// CHECK-NEXT:      04F0: 00000000 18000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      0500: 00017C1E 068B0000 00001B0C 1F000000  |..|.............|
// CHECK-NEXT:      0510: 10000000 20000000 00000000 04000000  |.... ...........|
// CHECK-NEXT:      0520: 00000000 1C000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:      0530: 00017C1E 0A8C0000 00000000 00001B0C  |..|.............|
// CHECK-NEXT:      0540: 1F000000 10000000 24000000 00000000  |........$.......|
// CHECK-NEXT:      0550: 04000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:      0560: 017A5052 00017C1E 0A900000 00000000  |.zPR..|.........|
// CHECK-NEXT:      0570: 00001B0C 1F000000 10000000 24000000  |............$...|
// CHECK-NEXT:      0580: 00000000 04000000 00000000 14000000  |................|
// CHECK-NEXT:      0590: 00000000 017A5052 00017C1E 04920000  |.....zPR..|.....|
// CHECK-NEXT:      05A0: 1B0C1F00 10000000 1C000000 00000000  |................|
// CHECK-NEXT:      05B0: 04000000 00000000 18000000 00000000  |................|
// CHECK-NEXT:      05C0: 017A5052 00017C1E 06930000 00001B0C  |.zPR..|.........|
// CHECK-NEXT:      05D0: 1F000000 10000000 20000000 00000000  |........ .......|
// CHECK-NEXT:      05E0: 04000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:      05F0: 017A5052 00017C1E 0A940000 00000000  |.zPR..|.........|
// CHECK-NEXT:      0600: 00001B0C 1F000000 10000000 24000000  |............$...|
// CHECK-NEXT:      0610: 00000000 04000000 00000000 1C000000  |................|
// CHECK-NEXT:      0620: 00000000 017A5052 00017C1E 0A980000  |.....zPR..|.....|
// CHECK-NEXT:      0630: 00000000 00001B0C 1F000000 10000000  |................|
// CHECK-NEXT:      0640: 24000000 00000000 04000000 00000000  |$...............|
// CHECK-NEXT:      0650: 14000000 00000000 017A5052 00017C1E  |.........zPR..|.|
// CHECK-NEXT:      0660: 049A0000 1B0C1F00 10000000 1C000000  |................|
// CHECK-NEXT:      0670: 00000000 04000000 00000000 18000000  |................|
// CHECK-NEXT:      0680: 00000000 017A5052 00017C1E 069B0000  |.....zPR..|.....|
// CHECK-NEXT:      0690: 00001B0C 1F000000 10000000 20000000  |............ ...|
// CHECK-NEXT:      06A0: 00000000 04000000 00000000 1C000000  |................|
// CHECK-NEXT:      06B0: 00000000 017A5052 00017C1E 0A9C0000  |.....zPR..|.....|
// CHECK-NEXT:      06C0: 00000000 00001B0C 1F000000 10000000  |................|
// CHECK-NEXT:      06D0: 24000000 00000000 04000000 00000000  |$...............|
// CHECK-NEXT:    )
// CHECK-NEXT:  }
// CHECK:       Section {
// CHECK:         Name: .rela.eh_frame (15)
// CHECK-NEXT:    Type: SHT_RELA (0x4)
// CHECK-NEXT:    Flags [ (0x40)
// CHECK-NEXT:      SHF_INFO_LINK (0x40)
// CHECK-NEXT:    ]
// CHECK-NEXT:    Address: 0x0
// CHECK-NEXT:    Offset: 0xBA0
// CHECK-NEXT:    Size: 1752
// CHECK-NEXT:    Link: 5
// CHECK-NEXT:    Info: 3
// CHECK-NEXT:    AddressAlignment: 8
// CHECK-NEXT:    EntrySize: 24
// CHECK-NEXT:    Relocations [
// CHECK-NEXT:      0x1C R_AARCH64_PREL32 .text 0x8C
// CHECK-NEXT:      0x48 R_AARCH64_PREL32 .text 0x8
// CHECK-NEXT:      0x51 R_AARCH64_ABS32 bar 0x0
// CHECK-NEXT:      0x78 R_AARCH64_PREL32 .text 0x0
// CHECK-NEXT:      0x81 R_AARCH64_ABS32 bar 0x0
// CHECK-NEXT:      0x9B R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0xB0 R_AARCH64_PREL32 .text 0xC
// CHECK-NEXT:      0xB9 R_AARCH64_ABS16 bar 0x0
// CHECK-NEXT:      0xCF R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0xE4 R_AARCH64_PREL32 .text 0x4
// CHECK-NEXT:      0xED R_AARCH64_ABS32 bar 0x0
// CHECK-NEXT:      0x106 R_AARCH64_ABS16 foo 0x0
// CHECK-NEXT:      0x114 R_AARCH64_PREL32 .text 0x10
// CHECK-NEXT:      0x132 R_AARCH64_ABS32 foo 0x0
// CHECK-NEXT:      0x144 R_AARCH64_PREL32 .text 0x14
// CHECK-NEXT:      0x162 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x178 R_AARCH64_PREL32 .text 0x18
// CHECK-NEXT:      0x196 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x1AC R_AARCH64_PREL32 .text 0x28
// CHECK-NEXT:      0x1CA R_AARCH64_ABS16 foo 0x0
// CHECK-NEXT:      0x1D8 R_AARCH64_PREL32 .text 0x1C
// CHECK-NEXT:      0x1F6 R_AARCH64_ABS32 foo 0x0
// CHECK-NEXT:      0x208 R_AARCH64_PREL32 .text 0x20
// CHECK-NEXT:      0x226 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x23C R_AARCH64_PREL32 .text 0x24
// CHECK-NEXT:      0x25A R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x270 R_AARCH64_PREL32 .text 0x2C
// CHECK-NEXT:      0x28E R_AARCH64_PREL16 foo 0x0
// CHECK-NEXT:      0x29C R_AARCH64_PREL32 .text 0x30
// CHECK-NEXT:      0x2BA R_AARCH64_PREL32 foo 0x0
// CHECK-NEXT:      0x2CC R_AARCH64_PREL32 .text 0x34
// CHECK-NEXT:      0x2EA R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x300 R_AARCH64_PREL32 .text 0x38
// CHECK-NEXT:      0x31E R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x334 R_AARCH64_PREL32 .text 0x48
// CHECK-NEXT:      0x352 R_AARCH64_PREL16 foo 0x0
// CHECK-NEXT:      0x360 R_AARCH64_PREL32 .text 0x3C
// CHECK-NEXT:      0x37E R_AARCH64_PREL32 foo 0x0
// CHECK-NEXT:      0x390 R_AARCH64_PREL32 .text 0x40
// CHECK-NEXT:      0x3AE R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x3C4 R_AARCH64_PREL32 .text 0x44
// CHECK-NEXT:      0x3E2 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x3F8 R_AARCH64_PREL32 .text 0x4C
// CHECK-NEXT:      0x416 R_AARCH64_ABS16 foo 0x0
// CHECK-NEXT:      0x424 R_AARCH64_PREL32 .text 0x50
// CHECK-NEXT:      0x442 R_AARCH64_ABS32 foo 0x0
// CHECK-NEXT:      0x454 R_AARCH64_PREL32 .text 0x54
// CHECK-NEXT:      0x472 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x488 R_AARCH64_PREL32 .text 0x58
// CHECK-NEXT:      0x4A6 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x4BC R_AARCH64_PREL32 .text 0x68
// CHECK-NEXT:      0x4DA R_AARCH64_ABS16 foo 0x0
// CHECK-NEXT:      0x4E8 R_AARCH64_PREL32 .text 0x5C
// CHECK-NEXT:      0x506 R_AARCH64_ABS32 foo 0x0
// CHECK-NEXT:      0x518 R_AARCH64_PREL32 .text 0x60
// CHECK-NEXT:      0x536 R_AARCH64_ABS64 foo 0x0
// CHECK-NEXT:      0x54C R_AARCH64_PREL32 .text 0x64
// CHECK-NEXT:      0x56A R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x580 R_AARCH64_PREL32 .text 0x6C
// CHECK-NEXT:      0x59E R_AARCH64_PREL16 foo 0x0
// CHECK-NEXT:      0x5AC R_AARCH64_PREL32 .text 0x70
// CHECK-NEXT:      0x5CA R_AARCH64_PREL32 foo 0x0
// CHECK-NEXT:      0x5DC R_AARCH64_PREL32 .text 0x74
// CHECK-NEXT:      0x5FA R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x610 R_AARCH64_PREL32 .text 0x78
// CHECK-NEXT:      0x62E R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x644 R_AARCH64_PREL32 .text 0x88
// CHECK-NEXT:      0x662 R_AARCH64_PREL16 foo 0x0
// CHECK-NEXT:      0x670 R_AARCH64_PREL32 .text 0x7C
// CHECK-NEXT:      0x68E R_AARCH64_PREL32 foo 0x0
// CHECK-NEXT:      0x6A0 R_AARCH64_PREL32 .text 0x80
// CHECK-NEXT:      0x6BE R_AARCH64_PREL64 foo 0x0
// CHECK-NEXT:      0x6D4 R_AARCH64_PREL32 .text 0x84
// CHECK-NEXT:    ]
// CHECK-NEXT:    SectionData (
// CHECK-NEXT:      0000: 1C000000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0010: 8C000000 00000000 48000000 00000000  |........H.......|
// CHECK-NEXT:      0020: 05010000 01000000 08000000 00000000  |................|
// CHECK-NEXT:      0030: 51000000 00000000 02010000 28000000  |Q...........(...|
// CHECK-NEXT:      0040: 00000000 00000000 78000000 00000000  |........x.......|
// CHECK-NEXT:      0050: 05010000 01000000 00000000 00000000  |................|
// CHECK-NEXT:      0060: 81000000 00000000 02010000 28000000  |............(...|
// CHECK-NEXT:      0070: 00000000 00000000 9B000000 00000000  |................|
// CHECK-NEXT:      0080: 01010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0090: B0000000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      00A0: 0C000000 00000000 B9000000 00000000  |................|
// CHECK-NEXT:      00B0: 03010000 28000000 00000000 00000000  |....(...........|
// CHECK-NEXT:      00C0: CF000000 00000000 01010000 29000000  |............)...|
// CHECK-NEXT:      00D0: 00000000 00000000 E4000000 00000000  |................|
// CHECK-NEXT:      00E0: 05010000 01000000 04000000 00000000  |................|
// CHECK-NEXT:      00F0: ED000000 00000000 02010000 28000000  |............(...|
// CHECK-NEXT:      0100: 00000000 00000000 06010000 00000000  |................|
// CHECK-NEXT:      0110: 03010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0120: 14010000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0130: 10000000 00000000 32010000 00000000  |........2.......|
// CHECK-NEXT:      0140: 02010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0150: 44010000 00000000 05010000 01000000  |D...............|
// CHECK-NEXT:      0160: 14000000 00000000 62010000 00000000  |........b.......|
// CHECK-NEXT:      0170: 01010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0180: 78010000 00000000 05010000 01000000  |x...............|
// CHECK-NEXT:      0190: 18000000 00000000 96010000 00000000  |................|
// CHECK-NEXT:      01A0: 01010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      01B0: AC010000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      01C0: 28000000 00000000 CA010000 00000000  |(...............|
// CHECK-NEXT:      01D0: 03010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      01E0: D8010000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      01F0: 1C000000 00000000 F6010000 00000000  |................|
// CHECK-NEXT:      0200: 02010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0210: 08020000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0220: 20000000 00000000 26020000 00000000  | .......&.......|
// CHECK-NEXT:      0230: 01010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0240: 3C020000 00000000 05010000 01000000  |<...............|
// CHECK-NEXT:      0250: 24000000 00000000 5A020000 00000000  |$.......Z.......|
// CHECK-NEXT:      0260: 04010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0270: 70020000 00000000 05010000 01000000  |p...............|
// CHECK-NEXT:      0280: 2C000000 00000000 8E020000 00000000  |,...............|
// CHECK-NEXT:      0290: 06010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      02A0: 9C020000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      02B0: 30000000 00000000 BA020000 00000000  |0...............|
// CHECK-NEXT:      02C0: 05010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      02D0: CC020000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      02E0: 34000000 00000000 EA020000 00000000  |4...............|
// CHECK-NEXT:      02F0: 04010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0300: 00030000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0310: 38000000 00000000 1E030000 00000000  |8...............|
// CHECK-NEXT:      0320: 04010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0330: 34030000 00000000 05010000 01000000  |4...............|
// CHECK-NEXT:      0340: 48000000 00000000 52030000 00000000  |H.......R.......|
// CHECK-NEXT:      0350: 06010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0360: 60030000 00000000 05010000 01000000  |`...............|
// CHECK-NEXT:      0370: 3C000000 00000000 7E030000 00000000  |<.......~.......|
// CHECK-NEXT:      0380: 05010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0390: 90030000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      03A0: 40000000 00000000 AE030000 00000000  |@...............|
// CHECK-NEXT:      03B0: 04010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      03C0: C4030000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      03D0: 44000000 00000000 E2030000 00000000  |D...............|
// CHECK-NEXT:      03E0: 01010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      03F0: F8030000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0400: 4C000000 00000000 16040000 00000000  |L...............|
// CHECK-NEXT:      0410: 03010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0420: 24040000 00000000 05010000 01000000  |$...............|
// CHECK-NEXT:      0430: 50000000 00000000 42040000 00000000  |P.......B.......|
// CHECK-NEXT:      0440: 02010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0450: 54040000 00000000 05010000 01000000  |T...............|
// CHECK-NEXT:      0460: 54000000 00000000 72040000 00000000  |T.......r.......|
// CHECK-NEXT:      0470: 01010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0480: 88040000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0490: 58000000 00000000 A6040000 00000000  |X...............|
// CHECK-NEXT:      04A0: 01010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      04B0: BC040000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      04C0: 68000000 00000000 DA040000 00000000  |h...............|
// CHECK-NEXT:      04D0: 03010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      04E0: E8040000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      04F0: 5C000000 00000000 06050000 00000000  |\...............|
// CHECK-NEXT:      0500: 02010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0510: 18050000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0520: 60000000 00000000 36050000 00000000  |`.......6.......|
// CHECK-NEXT:      0530: 01010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0540: 4C050000 00000000 05010000 01000000  |L...............|
// CHECK-NEXT:      0550: 64000000 00000000 6A050000 00000000  |d.......j.......|
// CHECK-NEXT:      0560: 04010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0570: 80050000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0580: 6C000000 00000000 9E050000 00000000  |l...............|
// CHECK-NEXT:      0590: 06010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      05A0: AC050000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      05B0: 70000000 00000000 CA050000 00000000  |p...............|
// CHECK-NEXT:      05C0: 05010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      05D0: DC050000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      05E0: 74000000 00000000 FA050000 00000000  |t...............|
// CHECK-NEXT:      05F0: 04010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0600: 10060000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      0610: 78000000 00000000 2E060000 00000000  |x...............|
// CHECK-NEXT:      0620: 04010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0630: 44060000 00000000 05010000 01000000  |D...............|
// CHECK-NEXT:      0640: 88000000 00000000 62060000 00000000  |........b.......|
// CHECK-NEXT:      0650: 06010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0660: 70060000 00000000 05010000 01000000  |p...............|
// CHECK-NEXT:      0670: 7C000000 00000000 8E060000 00000000  ||...............|
// CHECK-NEXT:      0680: 05010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      0690: A0060000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      06A0: 80000000 00000000 BE060000 00000000  |................|
// CHECK-NEXT:      06B0: 04010000 29000000 00000000 00000000  |....)...........|
// CHECK-NEXT:      06C0: D4060000 00000000 05010000 01000000  |................|
// CHECK-NEXT:      06D0: 84000000 00000000                    |........|
// CHECK-NEXT:    )
// CHECK-NEXT:  }

.ifdef ERR
// ERR: [[#@LINE+1]]:15: error: expected .eh_frame or .debug_frame
.cfi_sections $
// ERR: [[#@LINE+1]]:28: error: expected comma
.cfi_sections .debug_frame $
// ERR: [[#@LINE+1]]:39: error: expected comma
.cfi_sections .debug_frame, .eh_frame $

// ERR: [[#@LINE+1]]:16: error: unexpected token
.cfi_startproc $
// ERR: [[#@LINE+1]]:23: error: expected newline
.cfi_startproc simple $

// ERR: [[#@LINE+1]]:14: error: expected newline
.cfi_endproc $
.endif
