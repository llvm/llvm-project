// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --sr --sd | FileCheck %s

f1:
        .cfi_startproc
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

// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .eh_frame
// CHECK-NEXT:     Type: SHT_X86_64_UNWIND
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x68
// CHECK-NEXT:     Size: 1776
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:     ]
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 10000000 00000000 017A5200 01781001  |.........zR..x..|
// CHECK-NEXT:       0010: 1B000000 10000000 18000000 00000000  |................|
// CHECK-NEXT:       0020: 01000000 00000000 14000000 00000000  |................|
// CHECK-NEXT:       0030: 017A4C52 00017810 02031B0C 07089001  |.zLR..x.........|
// CHECK-NEXT:       0040: 14000000 1C000000 00000000 01000000  |................|
// CHECK-NEXT:       0050: 04000000 00000000 14000000 34000000  |............4...|
// CHECK-NEXT:       0060: 00000000 01000000 04000000 00000000  |................|
// CHECK-NEXT:       0070: 20000000 00000000 017A504C 52000178  | ........zPLR..x|
// CHECK-NEXT:       0080: 100B0000 00000000 00000002 1B0C0708  |................|
// CHECK-NEXT:       0090: 90010000 10000000 28000000 00000000  |........(.......|
// CHECK-NEXT:       00A0: 01000000 02000000 20000000 00000000  |........ .......|
// CHECK-NEXT:       00B0: 017A504C 52000178 100B0000 00000000  |.zPLR..x........|
// CHECK-NEXT:       00C0: 00000003 1B0C0708 90010000 14000000  |................|
// CHECK-NEXT:       00D0: 28000000 00000000 01000000 04000000  |(...............|
// CHECK-NEXT:       00E0: 00000000 18000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:       00F0: 00017810 04020000 1B0C0708 90010000  |..x.............|
// CHECK-NEXT:       0100: 10000000 20000000 00000000 01000000  |.... ...........|
// CHECK-NEXT:       0110: 00000000 18000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:       0120: 00017810 06030000 00001B0C 07089001  |..x.............|
// CHECK-NEXT:       0130: 10000000 20000000 00000000 01000000  |.... ...........|
// CHECK-NEXT:       0140: 00000000 1C000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:       0150: 00017810 0A040000 00000000 00001B0C  |..x.............|
// CHECK-NEXT:       0160: 07089001 10000000 24000000 00000000  |........$.......|
// CHECK-NEXT:       0170: 01000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:       0180: 017A5052 00017810 0A080000 00000000  |.zPR..x.........|
// CHECK-NEXT:       0190: 00001B0C 07089001 10000000 24000000  |............$...|
// CHECK-NEXT:       01A0: 00000000 01000000 00000000 18000000  |................|
// CHECK-NEXT:       01B0: 00000000 017A5052 00017810 040A0000  |.....zPR..x.....|
// CHECK-NEXT:       01C0: 1B0C0708 90010000 10000000 20000000  |............ ...|
// CHECK-NEXT:       01D0: 00000000 01000000 00000000 18000000  |................|
// CHECK-NEXT:       01E0: 00000000 017A5052 00017810 060B0000  |.....zPR..x.....|
// CHECK-NEXT:       01F0: 00001B0C 07089001 10000000 20000000  |............ ...|
// CHECK-NEXT:       0200: 00000000 01000000 00000000 1C000000  |................|
// CHECK-NEXT:       0210: 00000000 017A5052 00017810 0A0C0000  |.....zPR..x.....|
// CHECK-NEXT:       0220: 00000000 00001B0C 07089001 10000000  |................|
// CHECK-NEXT:       0230: 24000000 00000000 01000000 00000000  |$...............|
// CHECK-NEXT:       0240: 1C000000 00000000 017A5052 00017810  |.........zPR..x.|
// CHECK-NEXT:       0250: 0A100000 00000000 00001B0C 07089001  |................|
// CHECK-NEXT:       0260: 10000000 24000000 00000000 01000000  |....$...........|
// CHECK-NEXT:       0270: 00000000 18000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:       0280: 00017810 04120000 1B0C0708 90010000  |..x.............|
// CHECK-NEXT:       0290: 10000000 20000000 00000000 01000000  |.... ...........|
// CHECK-NEXT:       02A0: 00000000 18000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:       02B0: 00017810 06130000 00001B0C 07089001  |..x.............|
// CHECK-NEXT:       02C0: 10000000 20000000 00000000 01000000  |.... ...........|
// CHECK-NEXT:       02D0: 00000000 1C000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:       02E0: 00017810 0A140000 00000000 00001B0C  |..x.............|
// CHECK-NEXT:       02F0: 07089001 10000000 24000000 00000000  |........$.......|
// CHECK-NEXT:       0300: 01000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:       0310: 017A5052 00017810 0A180000 00000000  |.zPR..x.........|
// CHECK-NEXT:       0320: 00001B0C 07089001 10000000 24000000  |............$...|
// CHECK-NEXT:       0330: 00000000 01000000 00000000 18000000  |................|
// CHECK-NEXT:       0340: 00000000 017A5052 00017810 041A0000  |.....zPR..x.....|
// CHECK-NEXT:       0350: 1B0C0708 90010000 10000000 20000000  |............ ...|
// CHECK-NEXT:       0360: 00000000 01000000 00000000 18000000  |................|
// CHECK-NEXT:       0370: 00000000 017A5052 00017810 061B0000  |.....zPR..x.....|
// CHECK-NEXT:       0380: 00001B0C 07089001 10000000 20000000  |............ ...|
// CHECK-NEXT:       0390: 00000000 01000000 00000000 1C000000  |................|
// CHECK-NEXT:       03A0: 00000000 017A5052 00017810 0A1C0000  |.....zPR..x.....|
// CHECK-NEXT:       03B0: 00000000 00001B0C 07089001 10000000  |................|
// CHECK-NEXT:       03C0: 24000000 00000000 01000000 00000000  |$...............|
// CHECK-NEXT:       03D0: 1C000000 00000000 017A5052 00017810  |.........zPR..x.|
// CHECK-NEXT:       03E0: 0A800000 00000000 00001B0C 07089001  |................|
// CHECK-NEXT:       03F0: 10000000 24000000 00000000 01000000  |....$...........|
// CHECK-NEXT:       0400: 00000000 18000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:       0410: 00017810 04820000 1B0C0708 90010000  |..x.............|
// CHECK-NEXT:       0420: 10000000 20000000 00000000 01000000  |.... ...........|
// CHECK-NEXT:       0430: 00000000 18000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:       0440: 00017810 06830000 00001B0C 07089001  |..x.............|
// CHECK-NEXT:       0450: 10000000 20000000 00000000 01000000  |.... ...........|
// CHECK-NEXT:       0460: 00000000 1C000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:       0470: 00017810 0A840000 00000000 00001B0C  |..x.............|
// CHECK-NEXT:       0480: 07089001 10000000 24000000 00000000  |........$.......|
// CHECK-NEXT:       0490: 01000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:       04A0: 017A5052 00017810 0A880000 00000000  |.zPR..x.........|
// CHECK-NEXT:       04B0: 00001B0C 07089001 10000000 24000000  |............$...|
// CHECK-NEXT:       04C0: 00000000 01000000 00000000 18000000  |................|
// CHECK-NEXT:       04D0: 00000000 017A5052 00017810 048A0000  |.....zPR..x.....|
// CHECK-NEXT:       04E0: 1B0C0708 90010000 10000000 20000000  |............ ...|
// CHECK-NEXT:       04F0: 00000000 01000000 00000000 18000000  |................|
// CHECK-NEXT:       0500: 00000000 017A5052 00017810 068B0000  |.....zPR..x.....|
// CHECK-NEXT:       0510: 00001B0C 07089001 10000000 20000000  |............ ...|
// CHECK-NEXT:       0520: 00000000 01000000 00000000 1C000000  |................|
// CHECK-NEXT:       0530: 00000000 017A5052 00017810 0A8C0000  |.....zPR..x.....|
// CHECK-NEXT:       0540: 00000000 00001B0C 07089001 10000000  |................|
// CHECK-NEXT:       0550: 24000000 00000000 01000000 00000000  |$...............|
// CHECK-NEXT:       0560: 1C000000 00000000 017A5052 00017810  |.........zPR..x.|
// CHECK-NEXT:       0570: 0A900000 00000000 00001B0C 07089001  |................|
// CHECK-NEXT:       0580: 10000000 24000000 00000000 01000000  |....$...........|
// CHECK-NEXT:       0590: 00000000 18000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:       05A0: 00017810 04920000 1B0C0708 90010000  |..x.............|
// CHECK-NEXT:       05B0: 10000000 20000000 00000000 01000000  |.... ...........|
// CHECK-NEXT:       05C0: 00000000 18000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:       05D0: 00017810 06930000 00001B0C 07089001  |..x.............|
// CHECK-NEXT:       05E0: 10000000 20000000 00000000 01000000  |.... ...........|
// CHECK-NEXT:       05F0: 00000000 1C000000 00000000 017A5052  |.............zPR|
// CHECK-NEXT:       0600: 00017810 0A940000 00000000 00001B0C  |..x.............|
// CHECK-NEXT:       0610: 07089001 10000000 24000000 00000000  |........$.......|
// CHECK-NEXT:       0620: 01000000 00000000 1C000000 00000000  |................|
// CHECK-NEXT:       0630: 017A5052 00017810 0A980000 00000000  |.zPR..x.........|
// CHECK-NEXT:       0640: 00001B0C 07089001 10000000 24000000  |............$...|
// CHECK-NEXT:       0650: 00000000 01000000 00000000 18000000  |................|
// CHECK-NEXT:       0660: 00000000 017A5052 00017810 049A0000  |.....zPR..x.....|
// CHECK-NEXT:       0670: 1B0C0708 90010000 10000000 20000000  |............ ...|
// CHECK-NEXT:       0680: 00000000 01000000 00000000 18000000  |................|
// CHECK-NEXT:       0690: 00000000 017A5052 00017810 069B0000  |.....zPR..x.....|
// CHECK-NEXT:       06A0: 00001B0C 07089001 10000000 20000000  |............ ...|
// CHECK-NEXT:       06B0: 00000000 01000000 00000000 1C000000  |................|
// CHECK-NEXT:       06C0: 00000000 017A5052 00017810 0A9C0000  |.....zPR..x.....|
// CHECK-NEXT:       06D0: 00000000 00001B0C 07089001 10000000  |................|
// CHECK-NEXT:       06E0: 24000000 00000000 01000000 00000000  |$...............|

// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .rela.eh_frame
// CHECK-NEXT:     Type: SHT_RELA
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 1752
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 24
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:       0x1C R_X86_64_PC32 .text 0x23
// CHECK-NEXT:       0x48 R_X86_64_PC32 .text 0x0
// CHECK-NEXT:       0x51 R_X86_64_32 bar 0x0
// CHECK-NEXT:       0x60 R_X86_64_PC32 .text 0x2
// CHECK-NEXT:       0x69 R_X86_64_32 bar 0x0
// CHECK-NEXT:       0x83 R_X86_64_64 foo 0x0
// CHECK-NEXT:       0x9C R_X86_64_PC32 .text 0x3
// CHECK-NEXT:       0xA5 R_X86_64_16 bar 0x0
// CHECK-NEXT:       0xBB R_X86_64_64 foo 0x0
// CHECK-NEXT:       0xD4 R_X86_64_PC32 .text 0x1
// CHECK-NEXT:       0xDD R_X86_64_32 bar 0x0
// CHECK-NEXT:       0xF6 R_X86_64_16 foo 0x0
// CHECK-NEXT:       0x108 R_X86_64_PC32 .text 0x4
// CHECK-NEXT:       0x126 R_X86_64_32 foo 0x0
// CHECK-NEXT:       0x138 R_X86_64_PC32 .text 0x5
// CHECK-NEXT:       0x156 R_X86_64_64 foo 0x0
// CHECK-NEXT:       0x16C R_X86_64_PC32 .text 0x6
// CHECK-NEXT:       0x18A R_X86_64_64 foo 0x0
// CHECK-NEXT:       0x1A0 R_X86_64_PC32 .text 0xA
// CHECK-NEXT:       0x1BE R_X86_64_16 foo 0x0
// CHECK-NEXT:       0x1D0 R_X86_64_PC32 .text 0x7
// CHECK-NEXT:       0x1EE R_X86_64_32 foo 0x0
// CHECK-NEXT:       0x200 R_X86_64_PC32 .text 0x8
// CHECK-NEXT:       0x21E R_X86_64_64 foo 0x0
// CHECK-NEXT:       0x234 R_X86_64_PC32 .text 0x9
// CHECK-NEXT:       0x252 R_X86_64_PC64 foo 0x0
// CHECK-NEXT:       0x268 R_X86_64_PC32 .text 0xB
// CHECK-NEXT:       0x286 R_X86_64_PC16 foo 0x0
// CHECK-NEXT:       0x298 R_X86_64_PC32 .text 0xC
// CHECK-NEXT:       0x2B6 R_X86_64_PC32 foo 0x0
// CHECK-NEXT:       0x2C8 R_X86_64_PC32 .text 0xD
// CHECK-NEXT:       0x2E6 R_X86_64_PC64 foo 0x0
// CHECK-NEXT:       0x2FC R_X86_64_PC32 .text 0xE
// CHECK-NEXT:       0x31A R_X86_64_PC64 foo 0x0
// CHECK-NEXT:       0x330 R_X86_64_PC32 .text 0x12
// CHECK-NEXT:       0x34E R_X86_64_PC16 foo 0x0
// CHECK-NEXT:       0x360 R_X86_64_PC32 .text 0xF
// CHECK-NEXT:       0x37E R_X86_64_PC32 foo 0x0
// CHECK-NEXT:       0x390 R_X86_64_PC32 .text 0x10
// CHECK-NEXT:       0x3AE R_X86_64_PC64 foo 0x0
// CHECK-NEXT:       0x3C4 R_X86_64_PC32 .text 0x11
// CHECK-NEXT:       0x3E2 R_X86_64_64 foo 0x0
// CHECK-NEXT:       0x3F8 R_X86_64_PC32 .text 0x13
// CHECK-NEXT:       0x416 R_X86_64_16 foo 0x0
// CHECK-NEXT:       0x428 R_X86_64_PC32 .text 0x14
// CHECK-NEXT:       0x446 R_X86_64_32 foo 0x0
// CHECK-NEXT:       0x458 R_X86_64_PC32 .text 0x15
// CHECK-NEXT:       0x476 R_X86_64_64 foo 0x0
// CHECK-NEXT:       0x48C R_X86_64_PC32 .text 0x16
// CHECK-NEXT:       0x4AA R_X86_64_64 foo 0x0
// CHECK-NEXT:       0x4C0 R_X86_64_PC32 .text 0x1A
// CHECK-NEXT:       0x4DE R_X86_64_16 foo 0x0
// CHECK-NEXT:       0x4F0 R_X86_64_PC32 .text 0x17
// CHECK-NEXT:       0x50E R_X86_64_32 foo 0x0
// CHECK-NEXT:       0x520 R_X86_64_PC32 .text 0x18
// CHECK-NEXT:       0x53E R_X86_64_64 foo 0x0
// CHECK-NEXT:       0x554 R_X86_64_PC32 .text 0x19
// CHECK-NEXT:       0x572 R_X86_64_PC64 foo 0x0
// CHECK-NEXT:       0x588 R_X86_64_PC32 .text 0x1B
// CHECK-NEXT:       0x5A6 R_X86_64_PC16 foo 0x0
// CHECK-NEXT:       0x5B8 R_X86_64_PC32 .text 0x1C
// CHECK-NEXT:       0x5D6 R_X86_64_PC32 foo 0x0
// CHECK-NEXT:       0x5E8 R_X86_64_PC32 .text 0x1D
// CHECK-NEXT:       0x606 R_X86_64_PC64 foo 0x0
// CHECK-NEXT:       0x61C R_X86_64_PC32 .text 0x1E
// CHECK-NEXT:       0x63A R_X86_64_PC64 foo 0x0
// CHECK-NEXT:       0x650 R_X86_64_PC32 .text 0x22
// CHECK-NEXT:       0x66E R_X86_64_PC16 foo 0x0
// CHECK-NEXT:       0x680 R_X86_64_PC32 .text 0x1F
// CHECK-NEXT:       0x69E R_X86_64_PC32 foo 0x0
// CHECK-NEXT:       0x6B0 R_X86_64_PC32 .text 0x20
// CHECK-NEXT:       0x6CE R_X86_64_PC64 foo 0x0
// CHECK-NEXT:       0x6E4 R_X86_64_PC32 .text 0x21
// CHECK-NEXT:     ]
// CHECK:        }
