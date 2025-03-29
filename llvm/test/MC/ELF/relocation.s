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
        movabs  $0, %rax
        movabs  $0, %rax
        pushq    $bar
        movq	foo(%rip), %rdx
        leaq    foo-bar(%r14),%r14
        addq	$bar,%rax         # R_X86_64_32S
	.word   foo-bar
	.byte   foo-bar
	call foo

        leaq    -1+foo(%rip), %r11

        leaq  _GLOBAL_OFFSET_TABLE_(%rax), %r15
        leaq  _GLOBAL_OFFSET_TABLE_(%rip), %r15
        movl  $_GLOBAL_OFFSET_TABLE_, %eax
        movabs  $_GLOBAL_OFFSET_TABLE_, %rax

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
// CHECK-NEXT:       0x1 R_X86_64_32 .text 0x0
// CHECK-NEXT:       0x8 R_X86_64_32S .text 0x0
// CHECK-NEXT:       0x13 R_X86_64_32S .text 0x0
// CHECK-NEXT:       0x1A R_X86_64_32S .text 0x0
// CHECK-NEXT:       0x22 R_X86_64_32S .text 0x0
// CHECK-NEXT:       0x26 R_X86_64_32 .text 0x0
// CHECK-NEXT:       0x3F R_X86_64_32S .text 0x0
// CHECK-NEXT:       0x46 R_X86_64_PC32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x4D R_X86_64_PC32 foo 0x4D
// CHECK-NEXT:       0x54 R_X86_64_32S .text 0x0
// CHECK-NEXT:       0x58 R_X86_64_PC16 foo 0x58
// CHECK-NEXT:       0x5A R_X86_64_PC8 foo 0x5A
// CHECK-NEXT:       0x5C R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x63 R_X86_64_PC32 foo 0xFFFFFFFFFFFFFFFB
// CHECK-NEXT:       0x6A R_X86_64_GOTPC32 _GLOBAL_OFFSET_TABLE_ 0x3
// CHECK-NEXT:       0x71 R_X86_64_GOTPC32 _GLOBAL_OFFSET_TABLE_ 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:       0x76 R_X86_64_GOTPC32 _GLOBAL_OFFSET_TABLE_ 0x1
// CHECK-NEXT:       0x7C R_X86_64_GOTPC64 _GLOBAL_OFFSET_TABLE_ 0x2
// CHECK-NEXT:       0x84 R_X86_64_GOTPCREL foo 0x0
// CHECK-NEXT:       0x88 R_X86_64_PLT32 foo 0x0
// CHECK-NEXT:       0x9C R_X86_64_32 .text 0x9C
// CHECK-NEXT:       0xA1 R_X86_64_PC16 pr23771 0xFFFFFFFFFFFFFFFE
// CHECK-NEXT:       0xA3 R_X86_64_PC32 pr23272 0x0
// CHECK-NEXT:     ]
// CHECK-NEXT:   }

.ifdef ERR
// ERR: [[#@LINE+1]]:7: error: unsupported relocation type
.long foo@gotoff
.endif
