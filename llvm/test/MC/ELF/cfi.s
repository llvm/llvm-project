// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck %s

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

// CHECK:      # Section 4
// CHECK-NEXT: (('sh_name', 0x00000011) # '.eh_frame'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x0000000000000002)
// CHECK-NEXT:  ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:  ('sh_offset', 0x0000000000000068)
// CHECK-NEXT:  ('sh_size', 0x00000000000006c8)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x0000000000000008)
// CHECK-NEXT:  ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT:  ('_section_data', '14000000 00000000 017a4c52 00017810 02031b0c 07089001 14000000 1c000000 00000000 01000000 04000000 00000000 20000000 00000000 017a504c 52000178 100b0000 00000000 00000003 1b0c0708 90010000 14000000 28000000 00000000 01000000 04000000 00000000 14000000 70000000 00000000 01000000 04000000 00000000 20000000 00000000 017a504c 52000178 100b0000 00000000 00000002 1b0c0708 90010000 10000000 28000000 00000000 01000000 02000000 18000000 00000000 017a5052 00017810 04020000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 06030000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a040000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 040a0000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 060b0000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a0c0000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a080000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a100000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 04120000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 06130000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a140000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 041a0000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 061b0000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a1c0000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a180000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a800000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 04820000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 06830000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a840000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 048a0000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 068b0000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a8c0000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a880000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a900000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 04920000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 06930000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a940000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 049a0000 1b0c0708 90010000 10000000 20000000 00000000 01000000 00000000 18000000 00000000 017a5052 00017810 069b0000 00001b0c 07089001 10000000 20000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a9c0000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000 1c000000 00000000 017a5052 00017810 0a980000 00000000 00001b0c 07089001 10000000 24000000 00000000 01000000 00000000')
// CHECK-NEXT: ),

// CHECK:        # Section 5
// CHECK-NEXT: (('sh_name', 0x0000000c) # '.rela.eh_frame'
// CHECK-NEXT:  ('sh_type', 0x00000004)
// CHECK-NEXT:  ('sh_flags', 0x0000000000000000)
// CHECK-NEXT:  ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:  ('sh_offset', 0x0000000000000e30)
// CHECK-NEXT:  ('sh_size', 0x00000000000006c0)
// CHECK-NEXT:  ('sh_link', 0x00000007)
// CHECK-NEXT:  ('sh_info', 0x00000004)
// CHECK-NEXT:  ('sh_addralign', 0x0000000000000008)
// CHECK-NEXT:  ('sh_entsize', 0x0000000000000018)
// CHECK-NEXT:  ('_relocations', [
// CHECK-NEXT:   # Relocation 0
// CHECK-NEXT:   (('r_offset', 0x0000000000000020)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 1
// CHECK-NEXT:   (('r_offset', 0x0000000000000029)
// CHECK-NEXT:    ('r_sym', 0x00000028)
// CHECK-NEXT:    ('r_type', 0x0000000a)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 2
// CHECK-NEXT:   (('r_offset', 0x0000000000000043)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 3
// CHECK-NEXT:   (('r_offset', 0x000000000000005c)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000001)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 4
// CHECK-NEXT:   (('r_offset', 0x0000000000000065)
// CHECK-NEXT:    ('r_sym', 0x00000028)
// CHECK-NEXT:    ('r_type', 0x0000000a)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 5
// CHECK-NEXT:   (('r_offset', 0x0000000000000074)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000002)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 6
// CHECK-NEXT:   (('r_offset', 0x000000000000007d)
// CHECK-NEXT:    ('r_sym', 0x00000028)
// CHECK-NEXT:    ('r_type', 0x0000000a)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 7
// CHECK-NEXT:   (('r_offset', 0x0000000000000097)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 8
// CHECK-NEXT:   (('r_offset', 0x00000000000000b0)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000003)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 9
// CHECK-NEXT:   (('r_offset', 0x00000000000000b9)
// CHECK-NEXT:    ('r_sym', 0x00000028)
// CHECK-NEXT:    ('r_type', 0x0000000c)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 10
// CHECK-NEXT:   (('r_offset', 0x00000000000000ce)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x0000000c)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 11
// CHECK-NEXT:   (('r_offset', 0x00000000000000e0)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000004)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 12
// CHECK-NEXT:   (('r_offset', 0x00000000000000fe)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x0000000a)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 13
// CHECK-NEXT:   (('r_offset', 0x0000000000000110)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000005)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 14
// CHECK-NEXT:   (('r_offset', 0x000000000000012e)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 15
// CHECK-NEXT:   (('r_offset', 0x0000000000000144)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000006)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 16
// CHECK-NEXT:   (('r_offset', 0x0000000000000162)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x0000000c)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 17
// CHECK-NEXT:   (('r_offset', 0x0000000000000174)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000007)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 18
// CHECK-NEXT:   (('r_offset', 0x0000000000000192)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x0000000a)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 19
// CHECK-NEXT:   (('r_offset', 0x00000000000001a4)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000008)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 20
// CHECK-NEXT:   (('r_offset', 0x00000000000001c2)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 21
// CHECK-NEXT:   (('r_offset', 0x00000000000001d8)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000009)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 22
// CHECK-NEXT:   (('r_offset', 0x00000000000001f6)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 23
// CHECK-NEXT:   (('r_offset', 0x000000000000020c)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x000000000000000a)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 24
// CHECK-NEXT:   (('r_offset', 0x000000000000022a)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000018)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 25
// CHECK-NEXT:   (('r_offset', 0x0000000000000240)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x000000000000000b)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 26
// CHECK-NEXT:   (('r_offset', 0x000000000000025e)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x0000000d)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 27
// CHECK-NEXT:   (('r_offset', 0x0000000000000270)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x000000000000000c)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 28
// CHECK-NEXT:   (('r_offset', 0x000000000000028e)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 29
// CHECK-NEXT:   (('r_offset', 0x00000000000002a0)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x000000000000000d)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 30
// CHECK-NEXT:   (('r_offset', 0x00000000000002be)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000018)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 31
// CHECK-NEXT:   (('r_offset', 0x00000000000002d4)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x000000000000000e)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 32
// CHECK-NEXT:   (('r_offset', 0x00000000000002f2)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x0000000d)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 33
// CHECK-NEXT:   (('r_offset', 0x0000000000000304)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x000000000000000f)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 34
// CHECK-NEXT:   (('r_offset', 0x0000000000000322)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 35
// CHECK-NEXT:   (('r_offset', 0x0000000000000334)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000010)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 36
// CHECK-NEXT:   (('r_offset', 0x0000000000000352)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000018)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 37
// CHECK-NEXT:   (('r_offset', 0x0000000000000368)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000011)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 38
// CHECK-NEXT:   (('r_offset', 0x0000000000000386)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000018)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 39
// CHECK-NEXT:   (('r_offset', 0x000000000000039c)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000012)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 40
// CHECK-NEXT:   (('r_offset', 0x00000000000003ba)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 41
// CHECK-NEXT:   (('r_offset', 0x00000000000003d0)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000013)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 42
// CHECK-NEXT:   (('r_offset', 0x00000000000003ee)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x0000000c)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 43
// CHECK-NEXT:   (('r_offset', 0x0000000000000400)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000014)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 44
// CHECK-NEXT:   (('r_offset', 0x000000000000041e)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x0000000a)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 45
// CHECK-NEXT:   (('r_offset', 0x0000000000000430)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000015)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 46
// CHECK-NEXT:   (('r_offset', 0x000000000000044e)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 47
// CHECK-NEXT:   (('r_offset', 0x0000000000000464)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000016)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 48
// CHECK-NEXT:   (('r_offset', 0x0000000000000482)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x0000000c)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 49
// CHECK-NEXT:   (('r_offset', 0x0000000000000494)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000017)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 50
// CHECK-NEXT:   (('r_offset', 0x00000000000004b2)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x0000000a)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 51
// CHECK-NEXT:   (('r_offset', 0x00000000000004c4)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000018)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 52
// CHECK-NEXT:   (('r_offset', 0x00000000000004e2)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 53
// CHECK-NEXT:   (('r_offset', 0x00000000000004f8)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000019)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 54
// CHECK-NEXT:   (('r_offset', 0x0000000000000516)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000001)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 55
// CHECK-NEXT:   (('r_offset', 0x000000000000052c)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x000000000000001a)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 56
// CHECK-NEXT:   (('r_offset', 0x000000000000054a)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000018)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 57
// CHECK-NEXT:   (('r_offset', 0x0000000000000560)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x000000000000001b)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 58
// CHECK-NEXT:   (('r_offset', 0x000000000000057e)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x0000000d)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 59
// CHECK-NEXT:   (('r_offset', 0x0000000000000590)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x000000000000001c)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 60
// CHECK-NEXT:   (('r_offset', 0x00000000000005ae)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 61
// CHECK-NEXT:   (('r_offset', 0x00000000000005c0)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x000000000000001d)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 62
// CHECK-NEXT:   (('r_offset', 0x00000000000005de)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000018)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 63
// CHECK-NEXT:   (('r_offset', 0x00000000000005f4)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x000000000000001e)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 64
// CHECK-NEXT:   (('r_offset', 0x0000000000000612)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x0000000d)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 65
// CHECK-NEXT:   (('r_offset', 0x0000000000000624)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x000000000000001f)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 66
// CHECK-NEXT:   (('r_offset', 0x0000000000000642)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 67
// CHECK-NEXT:   (('r_offset', 0x0000000000000654)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000020)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 68
// CHECK-NEXT:   (('r_offset', 0x0000000000000672)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000018)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 69
// CHECK-NEXT:   (('r_offset', 0x0000000000000688)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000021)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 70
// CHECK-NEXT:   (('r_offset', 0x00000000000006a6)
// CHECK-NEXT:    ('r_sym', 0x00000029)
// CHECK-NEXT:    ('r_type', 0x00000018)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:   # Relocation 71
// CHECK-NEXT:   (('r_offset', 0x00000000000006bc)
// CHECK-NEXT:    ('r_sym', 0x00000024)
// CHECK-NEXT:    ('r_type', 0x00000002)
// CHECK-NEXT:    ('r_addend', 0x0000000000000022)
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])
// CHECK-NEXT: ),
