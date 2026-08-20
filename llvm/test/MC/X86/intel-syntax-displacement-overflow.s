# RUN: not llvm-mc -triple=x86_64 -x86-asm-syntax=intel %s 2>&1 | FileCheck %s --check-prefixes=CHECK,X64 --implicit-check-not=error: --implicit-check-not=warning:
# RUN: llvm-mc -triple=i686 -x86-asm-syntax=intel --defsym A16=1 %s 2>&1 | FileCheck %s --check-prefixes=CHECK,X86 --implicit-check-not=error: --implicit-check-not=warning:

.ifndef A16
mov rax, [rip+0x80000000-1]
lea rax, [rip-0x80000000]

# X64: [[#@LINE+1]]:10: error: displacement 2147483648 is not within [-2147483648, 2147483647]
mov rax, [rip+0x80000000]

# X64: [[#@LINE+1]]:10: error: displacement -2147483649 is not within [-2147483648, 2147483647]
lea rax, [rip-0x80000001]
.endif

mov eax, [eax+0xffffffff]
lea eax, [eax-0xffffffff]

# CHECK: [[#@LINE+1]]:10: warning: displacement 4294967296 shortened to 32-bit signed 0
mov eax, [eax+0xffffffff+1]

# CHECK: [[#@LINE+1]]:10: warning: displacement -4294967296 shortened to 32-bit signed 0
lea eax, [eax-0xffffffff-1]
# CHECK: [[#@LINE+1]]:10: warning: displacement -4294967297 shortened to 32-bit signed -1
lea eax, [eax-0xffffffff-2]

{disp8} lea eax, [ebx+0x100]
{disp8} lea eax, [ebx-0x100]

.ifdef A16
.code16
mov word ptr [bp+0xffff], 0
mov word ptr [si-0xffff], 0

# X86: [[#@LINE+1]]:14: warning: displacement 65536 shortened to 16-bit signed 0
mov word ptr [bp+0xffff+1], 0
# X86: [[#@LINE+1]]:14: warning: displacement -65536 shortened to 16-bit signed 0
mov word ptr [si-0xffff-1], 0
.endif
