# RUN: not llvm-mc -triple=x86_64 %s 2>&1 | FileCheck %s --check-prefixes=CHECK,X64 --implicit-check-not=error: --implicit-check-not=warning:
# RUN: llvm-mc -triple=i686 --defsym A16=1 %s 2>&1 | FileCheck %s --check-prefixes=CHECK,X86 --implicit-check-not=error: --implicit-check-not=warning:

.ifndef A16
movq 0x80000000-1(%rip), %rax
leaq -0x80000000(%rip), %rax

# X64: [[#@LINE+1]]:17: error: displacement 2147483648 is not within [-2147483648, 2147483647]
movq 0x80000000(%rip), %rax

# X64: [[#@LINE+1]]:18: error: displacement -2147483649 is not within [-2147483648, 2147483647]
leaq -0x80000001(%rip), %rax
.endif

movl 0xffffffff(%eax), %eax
leal -0xffffffff(%eax), %eax

# CHECK: [[#@LINE+1]]:19: warning: displacement 4294967296 shortened to 32-bit signed 0
movl 0xffffffff+1(%eax), %eax

# CHECK: [[#@LINE+1]]:20: warning: displacement -4294967296 shortened to 32-bit signed 0
leal -0xffffffff-1(%eax), %eax
# CHECK: [[#@LINE+1]]:20: warning: displacement -4294967297 shortened to 32-bit signed -1
leal -0xffffffff-2(%eax), %eax

{disp8} leal 0x100(%ebx), %eax
{disp8} leal -0x100(%ebx), %eax

.ifdef A16
.code16
movw $0, 0xffff(%bp)
movw $0, -0xffff(%si)

# X86: [[#@LINE+1]]:19: warning: displacement 65536 shortened to 16-bit signed 0
movw $0, 0xffff+1(%bp)
# X86: [[#@LINE+1]]:20: warning: displacement -65536 shortened to 16-bit signed 0
movw $0, -0xffff-1(%si)
.endif
