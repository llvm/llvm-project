# RUN: llvm-mc -filetype=obj -triple=x86_64 %s | llvm-readobj -r - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=x86_64 -relax-relocations=false %s | llvm-readobj -r - | FileCheck --check-prefix=NORELAX %s

# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.text {
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
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# NORELAX:      Relocations [
# NORELAX-NEXT:   Section ({{.*}}) .rela.text {
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
# NORELAX-NEXT:   }
# NORELAX-NEXT: ]

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
