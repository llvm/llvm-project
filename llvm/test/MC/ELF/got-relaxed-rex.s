// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o - | llvm-readobj -r - | FileCheck %s

// these should produce R_X86_64_REX_GOTPCRELX

        movq mov@GOTPCREL(%rip), %rax
        test %rax, test@GOTPCREL(%rip)
        adc adc@GOTPCREL(%rip), %rax
        add add@GOTPCREL(%rip), %rax
        and and@GOTPCREL(%rip), %rax
        cmp cmp@GOTPCREL(%rip), %rax
        or  or@GOTPCREL(%rip), %rax
        sbb sbb@GOTPCREL(%rip), %rax
        sub sub@GOTPCREL(%rip), %rax
        xor xor@GOTPCREL(%rip), %rax

.section .norelax,"ax"
## This expression loads the GOT entry with an offset.
## Don't emit R_X86_64_REX_GOTPCRELX.
        movq mov@GOTPCREL+1(%rip), %rax

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.text {
// CHECK-NEXT:     R_X86_64_REX_GOTPCRELX mov
// CHECK-NEXT:     R_X86_64_REX_GOTPCRELX test
// CHECK-NEXT:     R_X86_64_REX_GOTPCRELX adc
// CHECK-NEXT:     R_X86_64_REX_GOTPCRELX add
// CHECK-NEXT:     R_X86_64_REX_GOTPCRELX and
// CHECK-NEXT:     R_X86_64_REX_GOTPCRELX cmp
// CHECK-NEXT:     R_X86_64_REX_GOTPCRELX or
// CHECK-NEXT:     R_X86_64_REX_GOTPCRELX sbb
// CHECK-NEXT:     R_X86_64_REX_GOTPCRELX sub
// CHECK-NEXT:     R_X86_64_REX_GOTPCRELX xor
// CHECK-NEXT:   }
// CHECK-NEXT:   Section ({{.*}}) .rela.norelax {
// CHECK-NEXT:     R_X86_64_GOTPCREL mov
// CHECK-NEXT:   }
