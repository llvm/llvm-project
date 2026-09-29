# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck %s

# CHECK:     Relocations [
# CHECK-NEXT:  Section ({{.*}}) .rela.text {
# CHECK-NEXT:     R_X86_64_GOTPCREL mov
# CHECK-NEXT:     R_X86_64_GOTPCREL test
# CHECK-NEXT:     R_X86_64_GOTPCREL adc
# CHECK-NEXT:     R_X86_64_GOTPCREL add
# CHECK-NEXT:     R_X86_64_GOTPCREL and
# CHECK-NEXT:     R_X86_64_GOTPCREL cmp
# CHECK-NEXT:     R_X86_64_GOTPCREL or
# CHECK-NEXT:     R_X86_64_GOTPCREL sbb
# CHECK-NEXT:     R_X86_64_GOTPCREL sub
# CHECK-NEXT:     R_X86_64_GOTPCREL xor
# CHECK-NEXT:     R_X86_64_GOTPCREL call
# CHECK-NEXT:     R_X86_64_GOTPCREL jmp
# CHECK-NEXT:     R_X86_64_GOTPCREL mov
# CHECK-NEXT:     R_X86_64_GOTPCREL test
# CHECK-NEXT:     R_X86_64_GOTPCREL adc
# CHECK-NEXT:     R_X86_64_GOTPCREL add
# CHECK-NEXT:     R_X86_64_GOTPCREL and
# CHECK-NEXT:     R_X86_64_GOTPCREL cmp
# CHECK-NEXT:     R_X86_64_GOTPCREL or
# CHECK-NEXT:     R_X86_64_GOTPCREL sbb
# CHECK-NEXT:     R_X86_64_GOTPCREL sub
# CHECK-NEXT:     R_X86_64_GOTPCREL xor
# CHECK-NEXT:     R_X86_64_GOTPCREL mov
# CHECK-NEXT:     R_X86_64_GOTPCREL test
# CHECK-NEXT:     R_X86_64_GOTPCREL adc
# CHECK-NEXT:     R_X86_64_GOTPCREL add
# CHECK-NEXT:     R_X86_64_GOTPCREL and
# CHECK-NEXT:     R_X86_64_GOTPCREL cmp
# CHECK-NEXT:     R_X86_64_GOTPCREL or
# CHECK-NEXT:     R_X86_64_GOTPCREL sbb
# CHECK-NEXT:     R_X86_64_GOTPCREL sub
# CHECK-NEXT:     R_X86_64_GOTPCREL xor
# CHECK-NEXT:   }

movl mov@GOTPCREL_NORELAX(%rip), %eax
test %eax, test@GOTPCREL_NORELAX(%rip)
adc adc@GOTPCREL_NORELAX(%rip), %eax
add add@GOTPCREL_NORELAX(%rip), %eax
and and@GOTPCREL_NORELAX(%rip), %eax
cmp cmp@GOTPCREL_NORELAX(%rip), %eax
or  or@GOTPCREL_NORELAX(%rip), %eax
sbb sbb@GOTPCREL_NORELAX(%rip), %eax
sub sub@GOTPCREL_NORELAX(%rip), %eax
xor xor@GOTPCREL_NORELAX(%rip), %eax
call *call@GOTPCREL_NORELAX(%rip)
jmp *jmp@GOTPCREL_NORELAX(%rip)

movl mov@GOTPCREL_NORELAX(%rip), %r8d
test %r8d, test@GOTPCREL_NORELAX(%rip)
adc adc@GOTPCREL_NORELAX(%rip), %r8d
add add@GOTPCREL_NORELAX(%rip), %r8d
and and@GOTPCREL_NORELAX(%rip), %r8d
cmp cmp@GOTPCREL_NORELAX(%rip), %r8d
or  or@GOTPCREL_NORELAX(%rip), %r8d
sbb sbb@GOTPCREL_NORELAX(%rip), %r8d
sub sub@GOTPCREL_NORELAX(%rip), %r8d
xor xor@GOTPCREL_NORELAX(%rip), %r8d

movq mov@GOTPCREL_NORELAX(%rip), %rax
test %rax, test@GOTPCREL_NORELAX(%rip)
adc adc@GOTPCREL_NORELAX(%rip), %rax
add add@GOTPCREL_NORELAX(%rip), %rax
and and@GOTPCREL_NORELAX(%rip), %rax
cmp cmp@GOTPCREL_NORELAX(%rip), %rax
or  or@GOTPCREL_NORELAX(%rip), %rax
sbb sbb@GOTPCREL_NORELAX(%rip), %rax
sub sub@GOTPCREL_NORELAX(%rip), %rax
xor xor@GOTPCREL_NORELAX(%rip), %rax
