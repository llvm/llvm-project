// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --sr  - | FileCheck  %s
// RUN: not llvm-mc -filetype=obj -triple x86_64 --defsym ERR=1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

// Test that we produce the correct relocation.


        .section	.pr23272,"awG",@progbits,pr23272,comdat
	.globl pr23272
pr23272:
pr23272_2:
pr23272_3 = pr23272_2

        .text
bar:
        movl	$bar, %edx        # R_X86_64_32
        movq	$bar, %rdx        # R_X86_64_32S
        movq	$bar, bar(%rip)   # R_X86_64_32S
        movl	bar, %edx         # R_X86_64_32S
        movq	bar, %rdx         # R_X86_64_32S
.long bar                         # R_X86_64_32
        leaq	foo@GOTTPOFF(%rip), %rax # R_X86_64_GOTTPOFF
        movq    foo@GOTTPOFF(%rip), %r31 # R_X86_64_CODE_4_GOTTPOFF
        addq    foo@GOTTPOFF(%rip), %r31 # R_X86_64_CODE_4_GOTTPOFF
        # NDD
        addq %r8, foo@GOTTPOFF(%rip), %r16 # R_X86_64_CODE_6_GOTTPOFF
        addq foo@GOTTPOFF(%rip), %rax, %r12 # R_X86_64_CODE_6_GOTTPOFF
        # NDD + NF
        {nf} addq %r8, foo@GOTTPOFF(%rip), %r16 # R_X86_64_CODE_6_GOTTPOFF
        {nf} addq foo@GOTTPOFF(%rip), %rax, %r12 # R_X86_64_CODE_6_GOTTPOFF
        # NF
        {nf} addq foo@GOTTPOFF(%rip), %r12 # R_X86_64_CODE_6_GOTTPOFF

        leaq	foo@TLSGD(%rip), %rax    # R_X86_64_TLSGD
        leaq	foo@TPOFF(%rax), %rax    # R_X86_64_TPOFF32
        leaq	foo@TLSLD(%rip), %rdi    # R_X86_64_TLSLD
        leaq	foo@dtpoff(%rax), %rcx   # R_X86_64_DTPOFF32
        movabs  foo@GOT, %rax		 # R_X86_64_GOT64
        movabs  foo@GOTOFF, %rax	 # R_X86_64_GOTOFF64
        pushq    $bar
        movq	foo(%rip), %rdx
        leaq    foo-bar(%r14),%r14
        addq	$bar,%rax         # R_X86_64_32S
	.quad	foo@DTPOFF
        movabsq	$baz@TPOFF, %rax
	.word   foo-bar
	.byte   foo-bar
	call foo

        leaq    -1+foo(%rip), %r11

        leaq  _GLOBAL_OFFSET_TABLE_(%rax), %r15
        leaq  _GLOBAL_OFFSET_TABLE_(%rip), %r15
        movl  $_GLOBAL_OFFSET_TABLE_, %eax
        movabs  $_GLOBAL_OFFSET_TABLE_, %rax

        .quad    blah@SIZE                        # R_X86_64_SIZE64
        .quad    blah@SIZE + 32                   # R_X86_64_SIZE64
        .quad    blah@SIZE - 32                   # R_X86_64_SIZE64
         movl    blah@SIZE, %eax                  # R_X86_64_SIZE32
         movl    blah@SIZE + 32, %eax             # R_X86_64_SIZE32
         movl    blah@SIZE - 32, %eax             # R_X86_64_SIZE32

        .long   foo@gotpcrel
        .long foo@plt

        .quad	pr23272_2 - pr23272
        .quad	pr23272_3 - pr23272

	.global pr24486
pr24486:
	pr24486_alias = pr24486
	.long pr24486_alias

        .code16
        call pr23771

        .weak weak_sym
weak_sym:
        .long  pr23272-weak_sym

// CHECK:        Section {
// CHECK:          Name: .rela.text
// CHECK:          Relocations [
// CHECK-NEXT:       0x1 R_X86_64_32        .text
// CHECK-NEXT:       0x8 R_X86_64_32S       .text
// CHECK-NEXT:       0x13 R_X86_64_32S      .text
// CHECK-NEXT:       0x1A R_X86_64_32S      .text
// CHECK-NEXT:       0x22 R_X86_64_32S      .text
// CHECK-NEXT:       0x26 R_X86_64_32       .text
// CHECK-NEXT:       0x2D R_X86_64_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x35 R_X86_64_CODE_4_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x3D R_X86_64_CODE_4_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x47 R_X86_64_CODE_6_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x51 R_X86_64_CODE_6_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x5B R_X86_64_CODE_6_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x65 R_X86_64_CODE_6_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x6F R_X86_64_CODE_6_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x76 R_X86_64_TLSGD    foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x7D R_X86_64_TPOFF32  foo 0x0
// CHECK-NEXT:       0x84 R_X86_64_TLSLD    foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x8B R_X86_64_DTPOFF32 foo 0x0
// CHECK-NEXT:       0x91 R_X86_64_GOT64 foo 0x0
// CHECK-NEXT:       0x9B R_X86_64_GOTOFF64 foo 0x0
// CHECK-NEXT:       0xA4 R_X86_64_32S .text 0x0
// CHECK-NEXT:       0xAB R_X86_64_PC32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0xB2 R_X86_64_PC32 foo 0xB2
// CHECK-NEXT:       0xB9 R_X86_64_32S .text 0x0
// CHECK-NEXT:       0xBD R_X86_64_DTPOFF64 foo 0x0
// CHECK-NEXT:       0xC7 R_X86_64_TPOFF64 baz 0x0
// CHECK-NEXT:       0xCF R_X86_64_PC16 foo 0xCF
// CHECK-NEXT:       0xD1 R_X86_64_PC8 foo 0xD1
// CHECK-NEXT:       0xD3 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0xDA R_X86_64_PC32 foo 0xFFFFFFFFFFFFFFFB
// CHECK-NEXT:       0xE1 R_X86_64_GOTPC32 _GLOBAL_OFFSET_TABLE_ 0x3
// CHECK-NEXT:       0xE8 R_X86_64_GOTPC32 _GLOBAL_OFFSET_TABLE_ 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0xED R_X86_64_GOTPC32 _GLOBAL_OFFSET_TABLE_ 0x1
// CHECK-NEXT:       0xF3 R_X86_64_GOTPC64 _GLOBAL_OFFSET_TABLE_ 0x2
// CHECK-NEXT:       0xFB R_X86_64_SIZE64 blah 0x0
// CHECK-NEXT:       0x103 R_X86_64_SIZE64 blah 0x20
// CHECK-NEXT:       0x10B R_X86_64_SIZE64 blah 0xFFFFFFFFFFFFFFE0
// CHECK-NEXT:       0x116 R_X86_64_SIZE32 blah 0x0
// CHECK-NEXT:       0x11D R_X86_64_SIZE32 blah 0x20
// CHECK-NEXT:       0x124 R_X86_64_SIZE32 blah 0xFFFFFFFFFFFFFFE0
// CHECK-NEXT:       0x128 R_X86_64_GOTPCREL foo 0x0
// CHECK-NEXT:       0x12C R_X86_64_PLT32 foo 0x0
// CHECK-NEXT:       0x140 R_X86_64_32 .text 0x140
// CHECK-NEXT:       0x145 R_X86_64_PC16 pr23771 0xFFFFFFFFFFFFFFFE
// CHECK-NEXT:       0x147 R_X86_64_PC32 pr23272 0x0
// CHECK-NEXT:     ]
// CHECK-NEXT:   }

.ifdef ERR
// ERR: [[#@LINE+1]]:7: error: unsupported relocation type
.long foo@gotoff
.endif
