# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck %s --check-prefixes=CHECK,COMMON
# RUN: llvm-mc -filetype=obj -triple=x86_64 -relax-relocations=false %s -o %t1.o
# RUN: llvm-readobj -r %t1.o | FileCheck %s --check-prefixes=NORELAX,COMMON

# COMMON:     Relocations [
# COMMON-NEXT:  Section ({{.*}}) .rela.text {
# CHECK-NEXT:     R_X86_64_GOTPCRELX mov
# CHECK-NEXT:     R_X86_64_GOTPCRELX test
# CHECK-NEXT:     R_X86_64_GOTPCRELX adc
# CHECK-NEXT:     R_X86_64_GOTPCRELX add
# CHECK-NEXT:     R_X86_64_GOTPCRELX and
# CHECK-NEXT:     R_X86_64_GOTPCRELX cmp
# CHECK-NEXT:     R_X86_64_GOTPCRELX or
# CHECK-NEXT:     R_X86_64_GOTPCRELX sbb
# CHECK-NEXT:     R_X86_64_GOTPCRELX sub
# CHECK-NEXT:     R_X86_64_GOTPCRELX xor
# CHECK-NEXT:     R_X86_64_GOTPCRELX call
# CHECK-NEXT:     R_X86_64_GOTPCRELX jmp
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX mov
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX test
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX adc
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX add
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX and
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX cmp
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX or
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX sbb
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX sub
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX xor
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX mov
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX test
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX adc
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX add
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX and
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX cmp
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX or
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX sbb
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX sub
# CHECK-NEXT:     R_X86_64_REX_GOTPCRELX xor
# CHECK-NEXT:   }

# NORELAX-NEXT:     R_X86_64_GOTPCREL mov
# NORELAX-NEXT:     R_X86_64_GOTPCREL test
# NORELAX-NEXT:     R_X86_64_GOTPCREL adc
# NORELAX-NEXT:     R_X86_64_GOTPCREL add
# NORELAX-NEXT:     R_X86_64_GOTPCREL and
# NORELAX-NEXT:     R_X86_64_GOTPCREL cmp
# NORELAX-NEXT:     R_X86_64_GOTPCREL or
# NORELAX-NEXT:     R_X86_64_GOTPCREL sbb
# NORELAX-NEXT:     R_X86_64_GOTPCREL sub
# NORELAX-NEXT:     R_X86_64_GOTPCREL xor
# NORELAX-NEXT:     R_X86_64_GOTPCREL call
# NORELAX-NEXT:     R_X86_64_GOTPCREL jmp
# NORELAX-NEXT:     R_X86_64_GOTPCREL mov
# NORELAX-NEXT:     R_X86_64_GOTPCREL test
# NORELAX-NEXT:     R_X86_64_GOTPCREL adc
# NORELAX-NEXT:     R_X86_64_GOTPCREL add
# NORELAX-NEXT:     R_X86_64_GOTPCREL and
# NORELAX-NEXT:     R_X86_64_GOTPCREL cmp
# NORELAX-NEXT:     R_X86_64_GOTPCREL or
# NORELAX-NEXT:     R_X86_64_GOTPCREL sbb
# NORELAX-NEXT:     R_X86_64_GOTPCREL sub
# NORELAX-NEXT:     R_X86_64_GOTPCREL xor
# NORELAX-NEXT:     R_X86_64_GOTPCREL mov
# NORELAX-NEXT:     R_X86_64_GOTPCREL test
# NORELAX-NEXT:     R_X86_64_GOTPCREL adc
# NORELAX-NEXT:     R_X86_64_GOTPCREL add
# NORELAX-NEXT:     R_X86_64_GOTPCREL and
# NORELAX-NEXT:     R_X86_64_GOTPCREL cmp
# NORELAX-NEXT:     R_X86_64_GOTPCREL or
# NORELAX-NEXT:     R_X86_64_GOTPCREL sbb
# NORELAX-NEXT:     R_X86_64_GOTPCREL sub
# NORELAX-NEXT:     R_X86_64_GOTPCREL xor
# NORELAX-NEXT:   }

movl mov@GOTPCREL(%rip), %eax
test %eax, test@GOTPCREL(%rip)
adc adc@GOTPCREL(%rip), %eax
add add@GOTPCREL(%rip), %eax
and and@GOTPCREL(%rip), %eax
cmp cmp@GOTPCREL(%rip), %eax
or  or@GOTPCREL(%rip), %eax
sbb sbb@GOTPCREL(%rip), %eax
sub sub@GOTPCREL(%rip), %eax
xor xor@GOTPCREL(%rip), %eax
call *call@GOTPCREL(%rip)
jmp *jmp@GOTPCREL(%rip)

movl mov@GOTPCREL(%rip), %r8d
test %r8d, test@GOTPCREL(%rip)
adc adc@GOTPCREL(%rip), %r8d
add add@GOTPCREL(%rip), %r8d
and and@GOTPCREL(%rip), %r8d
cmp cmp@GOTPCREL(%rip), %r8d
or  or@GOTPCREL(%rip), %r8d
sbb sbb@GOTPCREL(%rip), %r8d
sub sub@GOTPCREL(%rip), %r8d
xor xor@GOTPCREL(%rip), %r8d

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

# COMMON-NEXT:   Section ({{.*}}) .rela.norelax {
# COMMON-NEXT:     R_X86_64_GOTPCREL mov 0x0
# COMMON-NEXT:     R_X86_64_GOTPCREL mov 0xFFFFFFFFFFFFFFFD
# COMMON-NEXT:     R_X86_64_GOTPCREL mov 0xFFFFFFFFFFFFFFFC
# COMMON-NEXT:   }
# COMMON-NEXT: ]

.section .norelax,"ax",@progbits
## Clang may emit this expression to load the high 32-bit of the GOT entry.
## Don't emit R_X86_64_GOTPCRELX.
movl mov@GOTPCREL+4(%rip), %eax
## Don't emit R_X86_64_GOTPCRELX.
movq mov@GOTPCREL+1(%rip), %rax
## We could emit R_X86_64_GOTPCRELX, but it is probably unnecessary.
movl mov@GOTPCREL+0(%rip), %eax
