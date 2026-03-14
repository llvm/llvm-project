// RUN: llvm-mc -triple x86_64-linux-gnu -filetype=obj %s | llvm-readobj -r - | FileCheck %s


				// CHECK:      Relocations [
				// CHECK-NEXT:   Section ({{[0-9]+}}) .rela.text {

pushq $foo			// CHECK-NEXT:     R_X86_64_32S
addq $foo, %rax			// CHECK-NEXT:     R_X86_64_32S
andq $foo, %rax			// CHECK-NEXT:     R_X86_64_32S
movq $foo, %rax			// CHECK-NEXT:     R_X86_64_32S
bextr $foo, (%edi), %eax	// CHECK-NEXT:     R_X86_64_32
bextr $foo, (%rdi), %rax	// CHECK-NEXT:     R_X86_64_32S
imul $foo, %rax			// CHECK-NEXT:     R_X86_64_32S

				// CHECK-NEXT:   }
				// CHECK-NEXT: ]
