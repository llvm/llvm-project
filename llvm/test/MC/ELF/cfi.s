// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -sr -sd | FileCheck %s

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
// CHECK-NEXT:       0000: 14000000 00000000 017A4C52 00017810
// CHECK-NEXT:       0010: 02031B0C 07089001 14000000 1C000000
// CHECK-NEXT:       0020: 00000000 01000000 04000000 00000000
// CHECK-NEXT:       0030: 20000000 00000000 017A504C 52000178
// CHECK-NEXT:       0040: 100B0000 00000000 00000003 1B0C0708
// CHECK-NEXT:       0050: 90010000 14000000 28000000 00000000
// CHECK-NEXT:       0060: 01000000 04000000 00000000 14000000
// CHECK-NEXT:       0070: 70000000 00000000 01000000 04000000
// CHECK-NEXT:       0080: 00000000 20000000 00000000 017A504C
// CHECK-NEXT:       0090: 52000178 100B0000 00000000 00000002
// CHECK-NEXT:       00A0: 1B0C0708 90010000 10000000 28000000
// CHECK-NEXT:       00B0: 00000000 01000000 02000000 18000000
// CHECK-NEXT:       00C0: 00000000 017A5052 00017810 04020000
// CHECK-NEXT:       00D0: 1B0C0708 90010000 10000000 20000000
// CHECK-NEXT:       00E0: 00000000 01000000 00000000 18000000
// CHECK-NEXT:       00F0: 00000000 017A5052 00017810 06030000
// CHECK-NEXT:       0100: 00001B0C 07089001 10000000 20000000
// CHECK-NEXT:       0110: 00000000 01000000 00000000 1C000000
// CHECK-NEXT:       0120: 00000000 017A5052 00017810 0A040000
// CHECK-NEXT:       0130: 00000000 00001B0C 07089001 10000000
// CHECK-NEXT:       0140: 24000000 00000000 01000000 00000000
// CHECK-NEXT:       0150: 18000000 00000000 017A5052 00017810
// CHECK-NEXT:       0160: 040A0000 1B0C0708 90010000 10000000
// CHECK-NEXT:       0170: 20000000 00000000 01000000 00000000
// CHECK-NEXT:       0180: 18000000 00000000 017A5052 00017810
// CHECK-NEXT:       0190: 060B0000 00001B0C 07089001 10000000
// CHECK-NEXT:       01A0: 20000000 00000000 01000000 00000000
// CHECK-NEXT:       01B0: 1C000000 00000000 017A5052 00017810
// CHECK-NEXT:       01C0: 0A0C0000 00000000 00001B0C 07089001
// CHECK-NEXT:       01D0: 10000000 24000000 00000000 01000000
// CHECK-NEXT:       01E0: 00000000 1C000000 00000000 017A5052
// CHECK-NEXT:       01F0: 00017810 0A080000 00000000 00001B0C
// CHECK-NEXT:       0200: 07089001 10000000 24000000 00000000
// CHECK-NEXT:       0210: 01000000 00000000 1C000000 00000000
// CHECK-NEXT:       0220: 017A5052 00017810 0A100000 00000000
// CHECK-NEXT:       0230: 00001B0C 07089001 10000000 24000000
// CHECK-NEXT:       0240: 00000000 01000000 00000000 18000000
// CHECK-NEXT:       0250: 00000000 017A5052 00017810 04120000
// CHECK-NEXT:       0260: 1B0C0708 90010000 10000000 20000000
// CHECK-NEXT:       0270: 00000000 01000000 00000000 18000000
// CHECK-NEXT:       0280: 00000000 017A5052 00017810 06130000
// CHECK-NEXT:       0290: 00001B0C 07089001 10000000 20000000
// CHECK-NEXT:       02A0: 00000000 01000000 00000000 1C000000
// CHECK-NEXT:       02B0: 00000000 017A5052 00017810 0A140000
// CHECK-NEXT:       02C0: 00000000 00001B0C 07089001 10000000
// CHECK-NEXT:       02D0: 24000000 00000000 01000000 00000000
// CHECK-NEXT:       02E0: 18000000 00000000 017A5052 00017810
// CHECK-NEXT:       02F0: 041A0000 1B0C0708 90010000 10000000
// CHECK-NEXT:       0300: 20000000 00000000 01000000 00000000
// CHECK-NEXT:       0310: 18000000 00000000 017A5052 00017810
// CHECK-NEXT:       0320: 061B0000 00001B0C 07089001 10000000
// CHECK-NEXT:       0330: 20000000 00000000 01000000 00000000
// CHECK-NEXT:       0340: 1C000000 00000000 017A5052 00017810
// CHECK-NEXT:       0350: 0A1C0000 00000000 00001B0C 07089001
// CHECK-NEXT:       0360: 10000000 24000000 00000000 01000000
// CHECK-NEXT:       0370: 00000000 1C000000 00000000 017A5052
// CHECK-NEXT:       0380: 00017810 0A180000 00000000 00001B0C
// CHECK-NEXT:       0390: 07089001 10000000 24000000 00000000
// CHECK-NEXT:       03A0: 01000000 00000000 1C000000 00000000
// CHECK-NEXT:       03B0: 017A5052 00017810 0A800000 00000000
// CHECK-NEXT:       03C0: 00001B0C 07089001 10000000 24000000
// CHECK-NEXT:       03D0: 00000000 01000000 00000000 18000000
// CHECK-NEXT:       03E0: 00000000 017A5052 00017810 04820000
// CHECK-NEXT:       03F0: 1B0C0708 90010000 10000000 20000000
// CHECK-NEXT:       0400: 00000000 01000000 00000000 18000000
// CHECK-NEXT:       0410: 00000000 017A5052 00017810 06830000
// CHECK-NEXT:       0420: 00001B0C 07089001 10000000 20000000
// CHECK-NEXT:       0430: 00000000 01000000 00000000 1C000000
// CHECK-NEXT:       0440: 00000000 017A5052 00017810 0A840000
// CHECK-NEXT:       0450: 00000000 00001B0C 07089001 10000000
// CHECK-NEXT:       0460: 24000000 00000000 01000000 00000000
// CHECK-NEXT:       0470: 18000000 00000000 017A5052 00017810
// CHECK-NEXT:       0480: 048A0000 1B0C0708 90010000 10000000
// CHECK-NEXT:       0490: 20000000 00000000 01000000 00000000
// CHECK-NEXT:       04A0: 18000000 00000000 017A5052 00017810
// CHECK-NEXT:       04B0: 068B0000 00001B0C 07089001 10000000
// CHECK-NEXT:       04C0: 20000000 00000000 01000000 00000000
// CHECK-NEXT:       04D0: 1C000000 00000000 017A5052 00017810
// CHECK-NEXT:       04E0: 0A8C0000 00000000 00001B0C 07089001
// CHECK-NEXT:       04F0: 10000000 24000000 00000000 01000000
// CHECK-NEXT:       0500: 00000000 1C000000 00000000 017A5052
// CHECK-NEXT:       0510: 00017810 0A880000 00000000 00001B0C
// CHECK-NEXT:       0520: 07089001 10000000 24000000 00000000
// CHECK-NEXT:       0530: 01000000 00000000 1C000000 00000000
// CHECK-NEXT:       0540: 017A5052 00017810 0A900000 00000000
// CHECK-NEXT:       0550: 00001B0C 07089001 10000000 24000000
// CHECK-NEXT:       0560: 00000000 01000000 00000000 18000000
// CHECK-NEXT:       0570: 00000000 017A5052 00017810 04920000
// CHECK-NEXT:       0580: 1B0C0708 90010000 10000000 20000000
// CHECK-NEXT:       0590: 00000000 01000000 00000000 18000000
// CHECK-NEXT:       05A0: 00000000 017A5052 00017810 06930000
// CHECK-NEXT:       05B0: 00001B0C 07089001 10000000 20000000
// CHECK-NEXT:       05C0: 00000000 01000000 00000000 1C000000
// CHECK-NEXT:       05D0: 00000000 017A5052 00017810 0A940000
// CHECK-NEXT:       05E0: 00000000 00001B0C 07089001 10000000
// CHECK-NEXT:       05F0: 24000000 00000000 01000000 00000000
// CHECK-NEXT:       0600: 18000000 00000000 017A5052 00017810
// CHECK-NEXT:       0610: 049A0000 1B0C0708 90010000 10000000
// CHECK-NEXT:       0620: 20000000 00000000 01000000 00000000
// CHECK-NEXT:       0630: 18000000 00000000 017A5052 00017810
// CHECK-NEXT:       0640: 069B0000 00001B0C 07089001 10000000
// CHECK-NEXT:       0650: 20000000 00000000 01000000 00000000
// CHECK-NEXT:       0660: 1C000000 00000000 017A5052 00017810
// CHECK-NEXT:       0670: 0A9C0000 00000000 00001B0C 07089001
// CHECK-NEXT:       0680: 10000000 24000000 00000000 01000000
// CHECK-NEXT:       0690: 00000000 1C000000 00000000 017A5052
// CHECK-NEXT:       06A0: 00017810 0A980000 00000000 00001B0C
// CHECK-NEXT:       06B0: 07089001 10000000 24000000 00000000
// CHECK-NEXT:       06C0: 01000000 00000000 10000000 00000000
// CHECK-NEXT:       06D0: 017A5200 01781001 1B000000 10000000
// CHECK-NEXT:       06E0: 18000000 00000000 01000000 00000000
// CHECK-NEXT:     )
// CHECK-NEXT:   }

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
// CHECK-NEXT:       0x20  R_X86_64_PC32 .text 0x0
// CHECK-NEXT:       0x29  R_X86_64_32   bar   0x0
// CHECK-NEXT:       0x43  R_X86_64_64   foo   0x0
// CHECK-NEXT:       0x5C  R_X86_64_PC32 .text 0x1
// CHECK-NEXT:       0x65  R_X86_64_32   bar   0x0
// CHECK-NEXT:       0x74  R_X86_64_PC32 .text 0x2
// CHECK-NEXT:       0x7D  R_X86_64_32   bar   0x0
// CHECK-NEXT:       0x97  R_X86_64_64   foo   0x0
// CHECK-NEXT:       0xB0  R_X86_64_PC32 .text 0x3
// CHECK-NEXT:       0xB9  R_X86_64_16   bar   0x0
// CHECK-NEXT:       0xCE  R_X86_64_16   foo   0x0
// CHECK-NEXT:       0xE0  R_X86_64_PC32 .text 0x4
// CHECK-NEXT:       0xFE  R_X86_64_32   foo   0x0
// CHECK-NEXT:       0x110 R_X86_64_PC32 .text 0x5
// CHECK-NEXT:       0x12E R_X86_64_64   foo   0x0
// CHECK-NEXT:       0x144 R_X86_64_PC32 .text 0x6
// CHECK-NEXT:       0x162 R_X86_64_16   foo   0x0
// CHECK-NEXT:       0x174 R_X86_64_PC32 .text 0x7
// CHECK-NEXT:       0x192 R_X86_64_32   foo   0x0
// CHECK-NEXT:       0x1A4 R_X86_64_PC32 .text 0x8
// CHECK-NEXT:       0x1C2 R_X86_64_64   foo   0x0
// CHECK-NEXT:       0x1D8 R_X86_64_PC32 .text 0x9
// CHECK-NEXT:       0x1F6 R_X86_64_64   foo   0x0
// CHECK-NEXT:       0x20C R_X86_64_PC32 .text 0xA
// CHECK-NEXT:       0x22A R_X86_64_PC64 foo   0x0
// CHECK-NEXT:       0x240 R_X86_64_PC32 .text 0xB
// CHECK-NEXT:       0x25E R_X86_64_PC16 foo   0x0
// CHECK-NEXT:       0x270 R_X86_64_PC32 .text 0xC
// CHECK-NEXT:       0x28E R_X86_64_PC32 foo   0x0
// CHECK-NEXT:       0x2A0 R_X86_64_PC32 .text 0xD
// CHECK-NEXT:       0x2BE R_X86_64_PC64 foo   0x0
// CHECK-NEXT:       0x2D4 R_X86_64_PC32 .text 0xE
// CHECK-NEXT:       0x2F2 R_X86_64_PC16 foo   0x0
// CHECK-NEXT:       0x304 R_X86_64_PC32 .text 0xF
// CHECK-NEXT:       0x322 R_X86_64_PC32 foo   0x0
// CHECK-NEXT:       0x334 R_X86_64_PC32 .text 0x10
// CHECK-NEXT:       0x352 R_X86_64_PC64 foo   0x0
// CHECK-NEXT:       0x368 R_X86_64_PC32 .text 0x11
// CHECK-NEXT:       0x386 R_X86_64_PC64 foo   0x0
// CHECK-NEXT:       0x39C R_X86_64_PC32 .text 0x12
// CHECK-NEXT:       0x3BA R_X86_64_64   foo   0x0
// CHECK-NEXT:       0x3D0 R_X86_64_PC32 .text 0x13
// CHECK-NEXT:       0x3EE R_X86_64_16   foo   0x0
// CHECK-NEXT:       0x400 R_X86_64_PC32 .text 0x14
// CHECK-NEXT:       0x41E R_X86_64_32   foo   0x0
// CHECK-NEXT:       0x430 R_X86_64_PC32 .text 0x15
// CHECK-NEXT:       0x44E R_X86_64_64   foo   0x0
// CHECK-NEXT:       0x464 R_X86_64_PC32 .text 0x16
// CHECK-NEXT:       0x482 R_X86_64_16   foo   0x0
// CHECK-NEXT:       0x494 R_X86_64_PC32 .text 0x17
// CHECK-NEXT:       0x4B2 R_X86_64_32   foo   0x0
// CHECK-NEXT:       0x4C4 R_X86_64_PC32 .text 0x18
// CHECK-NEXT:       0x4E2 R_X86_64_64   foo   0x0
// CHECK-NEXT:       0x4F8 R_X86_64_PC32 .text 0x19
// CHECK-NEXT:       0x516 R_X86_64_64   foo   0x0
// CHECK-NEXT:       0x52C R_X86_64_PC32 .text 0x1A
// CHECK-NEXT:       0x54A R_X86_64_PC64 foo   0x0
// CHECK-NEXT:       0x560 R_X86_64_PC32 .text 0x1B
// CHECK-NEXT:       0x57E R_X86_64_PC16 foo   0x0
// CHECK-NEXT:       0x590 R_X86_64_PC32 .text 0x1C
// CHECK-NEXT:       0x5AE R_X86_64_PC32 foo   0x0
// CHECK-NEXT:       0x5C0 R_X86_64_PC32 .text 0x1D
// CHECK-NEXT:       0x5DE R_X86_64_PC64 foo   0x0
// CHECK-NEXT:       0x5F4 R_X86_64_PC32 .text 0x1E
// CHECK-NEXT:       0x612 R_X86_64_PC16 foo   0x0
// CHECK-NEXT:       0x624 R_X86_64_PC32 .text 0x1F
// CHECK-NEXT:       0x642 R_X86_64_PC32 foo   0x0
// CHECK-NEXT:       0x654 R_X86_64_PC32 .text 0x20
// CHECK-NEXT:       0x672 R_X86_64_PC64 foo   0x0
// CHECK-NEXT:       0x688 R_X86_64_PC32 .text 0x21
// CHECK-NEXT:       0x6A6 R_X86_64_PC64 foo   0x0
// CHECK-NEXT:       0x6BC R_X86_64_PC32 .text 0x22
// CHECK-NEXT:       0x6E4 R_X86_64_PC32 .text 0x23
// CHECK-NEXT:     ]
// CHECK:        }
