// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -sr  | FileCheck  %s

// Test that we produce the correct relocation.


        .section	.pr23272,"aGw",@progbits,pr23272,comdat
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

        # this should probably be an error...
	zed = foo +2
	call zed@PLT

        leaq    -1+foo(%rip), %r11

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
// CHECK-NEXT:       0x34 R_X86_64_TLSGD    foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x3B R_X86_64_TPOFF32  foo 0x0
// CHECK-NEXT:       0x42 R_X86_64_TLSLD    foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x49 R_X86_64_DTPOFF32 foo 0x0
// CHECK-NEXT:       0x4F R_X86_64_GOT64 foo 0x0
// CHECK-NEXT:       0x59 R_X86_64_GOTOFF64 foo 0x0
// CHECK-NEXT:       0x62 R_X86_64_32S .text 0x0
// CHECK-NEXT:       0x69 R_X86_64_PC32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x70 R_X86_64_PC32 foo 0x70
// CHECK-NEXT:       0x77 R_X86_64_32S .text 0x0
// CHECK-NEXT:       0x7B R_X86_64_DTPOFF64 foo 0x0
// CHECK-NEXT:       0x85 R_X86_64_TPOFF64 baz 0x0
// CHECK-NEXT:       0x8D R_X86_64_PC16 foo 0x8D
// CHECK-NEXT:       0x8F R_X86_64_PC8 foo 0x8F
// CHECK-NEXT:       0x91 R_X86_64_PLT32 zed 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x98 R_X86_64_PC32 foo 0xFFFFFFFFFFFFFFFB
// CHECK-NEXT:       0x9D R_X86_64_GOTPC32 _GLOBAL_OFFSET_TABLE_ 0x1
// CHECK-NEXT:       0xA3 R_X86_64_GOTPC64 _GLOBAL_OFFSET_TABLE_ 0x2
// CHECK-NEXT:       0xAB R_X86_64_SIZE64 blah 0x0
// CHECK-NEXT:       0xB3 R_X86_64_SIZE64 blah 0x20
// CHECK-NEXT:       0xBB R_X86_64_SIZE64 blah 0xFFFFFFFFFFFFFFE0
// CHECK-NEXT:       0xC6 R_X86_64_SIZE32 blah 0x0
// CHECK-NEXT:       0xCD R_X86_64_SIZE32 blah 0x20
// CHECK-NEXT:       0xD4 R_X86_64_SIZE32 blah 0xFFFFFFFFFFFFFFE0
// CHECK-NEXT:       0xD8 R_X86_64_GOTPCREL foo 0x0
// CHECK-NEXT:       0xDC R_X86_64_PLT32 foo 0x0
// CHECK-NEXT:       0xF0 R_X86_64_32 .text 0xF0
// CHECK-NEXT:       0xF5 R_X86_64_PC16 pr23771 0xFFFFFFFFFFFFFFFE
// CHECK-NEXT:       0xF7 R_X86_64_PC32 pr23272 0x0
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
