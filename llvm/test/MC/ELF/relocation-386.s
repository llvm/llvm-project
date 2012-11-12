// RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o - | elf-dump | FileCheck  %s

// Test that we produce the correct relocation types and that the relocations
// correctly point to the section or the symbol.

// CHECK:      # Relocation 0
// CHECK-NEXT: (('r_offset', 0x00000002)
// CHECK-NEXT:  ('r_sym', 0x000001)
// CHECK-NEXT:  ('r_type', 0x09)
// CHECK-NEXT: ),
// CHECK-NEXT:  # Relocation 1
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym',
// CHECK-NEXT:  ('r_type', 0x04)
// CHECK-NEXT: ),
// CHECK-NEXT:  # Relocation 2
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym',
// CHECK-NEXT:  ('r_type', 0x0a)
// CHECK-NEXT: ),

// Relocation 3 (bar3@GOTOFF) is done with symbol 7 (bss)
// CHECK-NEXT:  # Relocation 3
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym', 0x000007
// CHECK-NEXT:  ('r_type',
// CHECK-NEXT: ),

// Relocation 4 (bar2@GOT) is of type R_386_GOT32
// CHECK-NEXT:  # Relocation 4
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym',
// CHECK-NEXT:  ('r_type', 0x03
// CHECK-NEXT: ),

// Relocation 5 (foo@TLSGD) is of type R_386_TLS_GD
// CHECK-NEXT: # Relocation 5
// CHECK-NEXT: (('r_offset', 0x00000020)
// CHECK-NEXT:  ('r_sym', 0x00000d)
// CHECK-NEXT:  ('r_type', 0x12)
// CHECK-NEXT: ),

// Relocation 6 ($foo@TPOFF) is of type R_386_TLS_LE_32
// CHECK-NEXT: # Relocation 6
// CHECK-NEXT: (('r_offset', 0x00000025)
// CHECK-NEXT:  ('r_sym', 0x00000d)
// CHECK-NEXT:  ('r_type', 0x22)
// CHECK-NEXT: ),

// Relocation 7 (foo@INDNTPOFF) is of type R_386_TLS_IE
// CHECK-NEXT: # Relocation 7
// CHECK-NEXT: (('r_offset', 0x0000002b)
// CHECK-NEXT:  ('r_sym', 0x00000d)
// CHECK-NEXT:  ('r_type', 0x0f)
// CHECK-NEXT: ),

// Relocation 8 (foo@NTPOFF) is of type R_386_TLS_LE
// CHECK-NEXT: # Relocation 8
// CHECK-NEXT: (('r_offset', 0x00000031)
// CHECK-NEXT:  ('r_sym', 0x00000d)
// CHECK-NEXT:  ('r_type', 0x11)
// CHECK-NEXT: ),

// Relocation 9 (foo@GOTNTPOFF) is of type R_386_TLS_GOTIE
// CHECK-NEXT: # Relocation 9
// CHECK-NEXT: (('r_offset', 0x00000037)
// CHECK-NEXT:  ('r_sym', 0x00000d)
// CHECK-NEXT:  ('r_type', 0x10)
// CHECK-NEXT: ),

// Relocation 10 (foo@TLSLDM) is of type R_386_TLS_LDM
// CHECK-NEXT: # Relocation 10
// CHECK-NEXT: (('r_offset', 0x0000003d)
// CHECK-NEXT:  ('r_sym', 0x00000d)
// CHECK-NEXT:  ('r_type', 0x13)
// CHECK-NEXT: ),

// Relocation 11 (foo@DTPOFF) is of type R_386_TLS_LDO_32
// CHECK-NEXT: # Relocation 11
// CHECK-NEXT: (('r_offset', 0x00000043)
// CHECK-NEXT:  ('r_sym', 0x00000d)
// CHECK-NEXT:  ('r_type', 0x20)
// CHECK-NEXT: ),
// Relocation 12 (calll 4096) is of type R_386_PC32
// CHECK-NEXT: # Relocation 12
// CHECK-NEXT: (('r_offset', 0x00000048)
// CHECK-NEXT:  ('r_sym', 0x000000)
// CHECK-NEXT:  ('r_type', 0x02)
// CHECK-NEXT: ),
// Relocation 13 (zed@GOT) is of type R_386_GOT32 and uses the symbol
// CHECK-NEXT: # Relocation 13
// CHECK-NEXT: (('r_offset', 0x0000004e)
// CHECK-NEXT:  ('r_sym', 0x000004)
// CHECK-NEXT:  ('r_type', 0x03)
// CHECK-NEXT: ),
// Relocation 14 (zed@GOTOFF) is of type R_386_GOTOFF and uses the symbol
// CHECK-NEXT: # Relocation 14
// CHECK-NEXT: (('r_offset', 0x00000054)
// CHECK-NEXT:  ('r_sym', 0x000004)
// CHECK-NEXT:  ('r_type', 0x09)
// CHECK-NEXT: ),
// Relocation 15 (zed@INDNTPOFF) is of type R_386_TLS_IE and uses the symbol
// CHECK-NEXT: # Relocation 15
// CHECK-NEXT: (('r_offset', 0x0000005a)
// CHECK-NEXT:  ('r_sym', 0x000004)
// CHECK-NEXT:  ('r_type', 0x0f)
// CHECK-NEXT: ),
// Relocation 16 (zed@NTPOFF) is of type R_386_TLS_LE and uses the symbol
// CHECK-NEXT: # Relocation 16
// CHECK-NEXT: (('r_offset', 0x00000060)
// CHECK-NEXT:  ('r_sym', 0x000004)
// CHECK-NEXT:  ('r_type', 0x11)
// CHECK-NEXT: ),
// Relocation 17 (zed@GOTNTPOFF) is of type R_386_TLS_GOTIE and uses the symbol
// CHECK-NEXT: # Relocation 17
// CHECK-NEXT: (('r_offset', 0x00000066)
// CHECK-NEXT:  ('r_sym', 0x000004)
// CHECK-NEXT:  ('r_type', 0x10)
// CHECK-NEXT: ),
// Relocation 18 (zed@PLT) is of type R_386_PLT32 and uses the symbol
// CHECK-NEXT: # Relocation 18
// CHECK-NEXT: (('r_offset', 0x0000006b)
// CHECK-NEXT:  ('r_sym', 0x000004)
// CHECK-NEXT:  ('r_type', 0x04)
// CHECK-NEXT: ),
// Relocation 19 (zed@TLSGD) is of type R_386_TLS_GD and uses the symbol
// CHECK-NEXT: # Relocation 19
// CHECK-NEXT: (('r_offset', 0x00000071)
// CHECK-NEXT:  ('r_sym', 0x000004)
// CHECK-NEXT:  ('r_type', 0x12)
// CHECK-NEXT: ),
// Relocation 20 (zed@TLSLDM) is of type R_386_TLS_LDM and uses the symbol
// CHECK-NEXT: # Relocation 20
// CHECK-NEXT: (('r_offset', 0x00000077)
// CHECK-NEXT:  ('r_sym', 0x000004)
// CHECK-NEXT:  ('r_type', 0x13)
// CHECK-NEXT: ),
// Relocation 21 (zed@TPOFF) is of type R_386_TLS_LE_32 and uses the symbol
// CHECK-NEXT:# Relocation 21
// CHECK-NEXT: (('r_offset', 0x0000007d)
// CHECK-NEXT:  ('r_sym', 0x000004)
// CHECK-NEXT:  ('r_type', 0x22)
// CHECK-NEXT: ),
// Relocation 22 (zed@DTPOFF) is of type R_386_TLS_LDO_32 and uses the symbol
// CHECK-NEXT: Relocation 22
// CHECK-NEXT: (('r_offset', 0x00000083)
// CHECK-NEXT:  ('r_sym', 0x000004)
// CHECK-NEXT:  ('r_type', 0x20)
// CHECK-NEXT: ),
// Relocation 23 ($bar) is of type R_386_32 and uses the section
// CHECK-NEXT: Relocation 23
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym',
// CHECK-NEXT:  ('r_type', 0x01)
// CHECK-NEXT: ),
// Relocation 24 (foo@GOTTPOFF(%edx)) is of type R_386_TLS_IE_32 and uses the
// symbol
// CHECK-NEXT: Relocation 24
// CHECK-NEXT: (('r_offset', 0x0000008e)
// CHECK-NEXT:  ('r_sym', 0x00000d)
// CHECK-NEXT:  ('r_type', 0x21)
// CHECK-NEXT: ),
// Relocation 25 (_GLOBAL_OFFSET_TABLE_-bar2) is of type R_386_GOTPC.
// CHECK-NEXT: Relocation 25
// CHECK-NEXT: (('r_offset', 0x00000094)
// CHECK-NEXT:  ('r_sym', 0x00000b)
// CHECK-NEXT:  ('r_type', 0x0a)
// CHECK-NEXT: ),
// Relocation 26 (und_symbol-bar2) is of type R_386_PC32
// CHECK-NEXT: Relocation 26
// CHECK-NEXT: (('r_offset', 0x0000009a)
// CHECK-NEXT:  ('r_sym', 0x00000e)
// CHECK-NEXT:  ('r_type', 0x02)
// CHECK-NEXT: ),

// Section 4 is bss
// CHECK:      # Section 4
// CHECK-NEXT: (('sh_name', 0x0000000b) # '.bss'

// CHECK:      # Symbol 1
// CHECK-NEXT: (('st_name', 0x00000005) # '.Lfoo'

// Symbol 4 is zed
// CHECK:      # Symbol 4
// CHECK-NEXT: (('st_name', 0x00000035) # 'zed'
// CHECK-NEXT:  ('st_value', 0x00000000)
// CHECK-NEXT:  ('st_size', 0x00000000)
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x6)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0005)

// Symbol 7 is section 4
// CHECK:      # Symbol 7
// CHECK-NEXT: (('st_name', 0x00000000) # ''
// CHECK-NEXT:  ('st_value', 0x00000000)
// CHECK-NEXT:  ('st_size', 0x00000000)
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x3)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0004)


        .text
bar:
	leal	.Lfoo@GOTOFF(%ebx), %eax

        .global bar2
bar2:
	calll	bar2@PLT
	addl	$_GLOBAL_OFFSET_TABLE_, %ebx
	movb	bar3@GOTOFF(%ebx), %al

	.type	bar3,@object
	.local	bar3
	.comm	bar3,1,1

        movl	bar2j@GOT(%eax), %eax

        leal foo@TLSGD(, %ebx,1), %eax
        movl $foo@TPOFF, %edx
        movl foo@INDNTPOFF, %ecx
        addl foo@NTPOFF(%eax), %eax
        addl foo@GOTNTPOFF(%ebx), %ecx
        leal foo@TLSLDM(%ebx), %eax
        leal foo@DTPOFF(%eax), %edx
        calll 4096
        movl zed@GOT(%eax), %eax
        movl zed@GOTOFF(%eax), %eax
        movl zed@INDNTPOFF(%eax), %eax
        movl zed@NTPOFF(%eax), %eax
        movl zed@GOTNTPOFF(%eax), %eax
        call zed@PLT
        movl zed@TLSGD(%eax), %eax
        movl zed@TLSLDM(%eax), %eax
        movl zed@TPOFF(%eax), %eax
        movl zed@DTPOFF(%eax), %eax
        pushl $bar
        addl foo@GOTTPOFF(%edx), %eax
        subl    _GLOBAL_OFFSET_TABLE_-bar2, %ebx
        leal und_symbol-bar2(%edx),%ecx

        .section        zedsec,"awT",@progbits
zed:
        .long 0

        .section	.rodata.str1.16,"aMS",@progbits,1
.Lfoo:
	.asciz	 "bool llvm::llvm_start_multithreaded()"
