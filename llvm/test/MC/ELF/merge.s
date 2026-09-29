// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r - | FileCheck  %s

// Test that relocations with local symbols in a mergeable section are done
// with a reference to the symbol if the offset is non zero.


        movsd   .Lfoo(%rip), %xmm1
        movl	$.Lfoo, %edi
        movl	$.Lfoo+2, %edi
        jmp	foo@PLT
        movq 	foo@GOTPCREL, %rax
        movq    zed, %rax
        movsd   .Lfoo+4(%rip), %xmm1

        .section        .sec1,"aM",@progbits,16
.Lfoo:
zed:
        .global zed

        .section	bar,"ax",@progbits
foo:

// CHECK:      Relocations [
// CHECK-NEXT:   Section {{.*}} .rela.text {
// CHECK-NEXT:     0x{{[^ ]+}} R_X86_64_PC32    .Lfoo 0x{{[^ ]+}}
// CHECK-NEXT:     0x{{[^ ]+}} R_X86_64_32      .sec1 0x{{[^ ]+}}
// CHECK-NEXT:     0x{{[^ ]+}} R_X86_64_32      .Lfoo 0x{{[^ ]+}}
// CHECK-NEXT:     0x{{[^ ]+}} R_X86_64_PLT32    foo  0x{{[^ ]+}}
// CHECK-NEXT:     0x{{[^ ]+}} R_X86_64_GOTPCREL foo  0x{{[^ ]+}}
// CHECK-NEXT:     0x{{[^ ]+}} R_X86_64_32S      zed  0x{{[^ ]+}}
// CHECK-NEXT:     0x{{[^ ]+}} R_X86_64_PC32    .sec1 0x{{[^ ]+}}
// CHECK-NEXT:   }
// CHECK-NEXT: ]
