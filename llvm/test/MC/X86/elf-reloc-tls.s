# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# CHECK:      .rela.GOTTPOFF {
# CHECK-NEXT:   0x3 R_X86_64_GOTTPOFF tls 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:   0xB R_X86_64_CODE_4_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:   0x13 R_X86_64_CODE_4_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:   0x1D R_X86_64_CODE_6_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:   0x27 R_X86_64_CODE_6_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:   0x31 R_X86_64_CODE_6_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:   0x3B R_X86_64_CODE_6_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:   0x45 R_X86_64_CODE_6_GOTTPOFF foo 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT: }

.section .TPOFF,"ax"
leaq	foo@TPOFF(%rax), %rax    # R_X86_64_TPOFF32
movabsq	$baz@TPOFF, %rax

.section .GOTTPOFF,"ax"
leaq tls@GOTTPOFF(%rip), %rax

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

.section .GD,"ax"
leaq	foo@TLSGD(%rip), %rax    # R_X86_64_TLSGD
leaq	foo@TLSLD(%rip), %rdi    # R_X86_64_TLSLD
leaq	foo@dtpoff(%rax), %rcx   # R_X86_64_DTPOFF32
.quad	foo@DTPOFF
