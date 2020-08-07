// RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -relax-relocations=false -o - | llvm-readobj -r  - | FileCheck  %s --check-prefix=CHECK --check-prefix=I386
// RUN: llvm-mc -filetype=obj -triple i386-pc-elfiamcu %s -relax-relocations=false  -o - | llvm-readobj -r  - | FileCheck  %s --check-prefix=CHECK --check-prefix=IAMCU

// Test that we produce the correct relocation types and that the relocations
// correctly point to the section or the symbol.

// IAMCU: Format: elf32-iamcu
// I386: Format: elf32-i386
// CHECK:      Relocations [
// CHECK-NEXT:   Section {{.*}} .rel.text {
// CHECK-NEXT:     0x2          R_386_GOTOFF     .Lfoo 0x0
// CHECK-NEXT:     0x{{[^ ]+}}  R_386_PLT32      bar2 0x0
// CHECK-NEXT:     0x{{[^ ]+}}  R_386_GOTPC      _GLOBAL_OFFSET_TABLE_ 0x0
// Relocation 3 (bar3@GOTOFF) is done with symbol 7 (bss)
// CHECK-NEXT:     0x{{[^ ]+}}  R_386_GOTOFF     .bss 0x0
// Relocation 4 (bar2@GOT) is of type R_386_GOT32
// CHECK-NEXT:     0x{{[^ ]+}}  R_386_GOT32      bar2j 0x0

// Relocation 5 (foo@TLSGD) is of type R_386_TLS_GD
// CHECK-NEXT:     0x20         R_386_TLS_GD     foo 0x0
// Relocation 6 ($foo@TPOFF) is of type R_386_TLS_LE_32
// CHECK-NEXT:     0x25         R_386_TLS_LE_32  foo 0x0
// Relocation 7 (foo@INDNTPOFF) is of type R_386_TLS_IE
// CHECK-NEXT:     0x2B         R_386_TLS_IE     foo 0x0
// Relocation 8 (foo@NTPOFF) is of type R_386_TLS_LE
// CHECK-NEXT:     0x31         R_386_TLS_LE     foo 0x0
// Relocation 9 (foo@GOTNTPOFF) is of type R_386_TLS_GOTIE
// CHECK-NEXT:     0x37         R_386_TLS_GOTIE  foo 0x0
// Relocation 10 (foo@TLSLDM) is of type R_386_TLS_LDM
// CHECK-NEXT:     0x3D         R_386_TLS_LDM    foo 0x0
// Relocation 11 (foo@DTPOFF) is of type R_386_TLS_LDO_32
// CHECK-NEXT:     0x43         R_386_TLS_LDO_32 foo 0x0
// Relocation 12 (calll 4096) is of type R_386_PC32
// CHECK-NEXT:     0x48         R_386_PC32       - 0x0
// Relocation 13 (zed@GOT) is of type R_386_GOT32 and uses the symbol
// CHECK-NEXT:     0x4E         R_386_GOT32      zed 0x0
// Relocation 14 (zed@GOTOFF) is of type R_386_GOTOFF and uses the symbol
// CHECK-NEXT:     0x54         R_386_GOTOFF     zed 0x0
// Relocation 15 (zed@INDNTPOFF) is of type R_386_TLS_IE and uses the symbol
// CHECK-NEXT:     0x5A         R_386_TLS_IE     zed 0x0
// Relocation 16 (zed@NTPOFF) is of type R_386_TLS_LE and uses the symbol
// CHECK-NEXT:     0x60         R_386_TLS_LE     zed 0x0
// Relocation 17 (zed@GOTNTPOFF) is of type R_386_TLS_GOTIE and uses the symbol
// CHECK-NEXT:     0x66         R_386_TLS_GOTIE  zed 0x0
// Relocation 18 (zed@PLT) is of type R_386_PLT32 and uses the symbol
// CHECK-NEXT:     0x6B         R_386_PLT32      zed 0x0
// Relocation 19 (zed@TLSGD) is of type R_386_TLS_GD and uses the symbol
// CHECK-NEXT:     0x71         R_386_TLS_GD     zed 0x0
// Relocation 20 (zed@TLSLDM) is of type R_386_TLS_LDM and uses the symbol
// CHECK-NEXT:     0x77         R_386_TLS_LDM    zed 0x0
// Relocation 21 (zed@TPOFF) is of type R_386_TLS_LE_32 and uses the symbol
// CHECK-NEXT:     0x7D         R_386_TLS_LE_32  zed 0x0
// Relocation 22 (zed@DTPOFF) is of type R_386_TLS_LDO_32 and uses the symbol
// CHECK-NEXT:     0x83         R_386_TLS_LDO_32 zed 0x0
// Relocation 23 ($bar) is of type R_386_32 and uses the section
// CHECK-NEXT:     0x{{[^ ]+}}  R_386_32         .text 0x0
// Relocation 24 (foo@GOTTPOFF(%edx)) is of type R_386_TLS_IE_32 and uses the
// symbol
// CHECK-NEXT:     0x8E         R_386_TLS_IE_32  foo 0x0
// Relocation 25 (_GLOBAL_OFFSET_TABLE_-bar2) is of type R_386_GOTPC.
// CHECK-NEXT:     0x94         R_386_GOTPC      _GLOBAL_OFFSET_TABLE_ 0x0
// Relocation 26 (und_symbol-bar2) is of type R_386_PC32
// CHECK-NEXT:     0x9A         R_386_PC32       und_symbol 0x0
// Relocation 27 (und_symbol-bar2) is of type R_386_PC16
// CHECK-NEXT:     0x9E         R_386_PC16       und_symbol 0x0
// Relocation 28 (und_symbol-bar2) is of type R_386_PC8
// CHECK-NEXT:     0xA0         R_386_PC8        und_symbol 0x0
// CHECK-NEXT:     0xA3         R_386_GOTOFF     und_symbol 0x0
// Relocation 29 (zed@PLT) is of type R_386_PLT32 and uses the symbol
// CHECK-NEXT:     0xA9         R_386_PLT32      zed 0x0
// CHECK-NEXT:     0xAF         R_386_PC32       tr_start 0x0
// CHECK-NEXT:     0xB3         R_386_16         foo      0x0
// CHECK-NEXT:     0xB5         R_386_8          foo      0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

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
        .word und_symbol-bar2
        .byte und_symbol-bar2

        leal 1 + und_symbol@GOTOFF, %edi
        movl zed@PLT(%eax), %eax

        .code64
        jmpq *tr_start(%rip)

        .word foo
        .byte foo

        .section        zedsec,"awT",@progbits
zed:
        .long 0

        .section	.rodata.str1.16,"aMS",@progbits,1
.Lfoo:
	.asciz	 "bool llvm::llvm_start_multithreaded()"
