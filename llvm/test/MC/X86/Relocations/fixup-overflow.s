# RUN: not llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## Test fixup value overflow diagnostics for non-PC-relative fixups
## with large immediate values that don't fit in the fixup field.

.intel_syntax noprefix
# CHECK: :[[#@LINE+1]]:10: error: value of 4294967296 is too large for field of 4 bytes
mov rcx, s4294967296

## GAS rejects [2147483648, 4294967295] for R_X86_64_32S
mov rcx, s4294967295

.long s4294967295
# CHECK: :[[#@LINE+1]]:7: error: value of 4294967296 is too large for field of 4 bytes
.long s4294967296

.quad s4294967296

# CHECK: :[[#@LINE+1]]:10: error: value of -4294967296 is too large for field of 4 bytes
mov rcx, sn4294967296

## GAS rejects [-4294967295, -2147483649] for R_X86_64_32S
mov rcx, sn4294967295

.long sn4294967295
# CHECK: :[[#@LINE+1]]:7: error: value of -4294967296 is too large for field of 4 bytes
.long sn4294967296

.set s4294967295, (1<<32)-1
.set s4294967296, 1<<32
.set sn4294967295, -(1<<32)+1
.set sn4294967296, -(1<<32)

.section rip_relative_fixups,"ax",@progbits
## RIP-relative addressing uses signed 32-bit displacement.
## Forward: 0x7fffffff fits, 0x80000000 overflows
mov eax, [rip + .Lend1]
.space 0x7fffffff
.Lend1:

# CHECK: :[[#@LINE+1]]:17: error: value of 2147483648 is too large for field of 4 bytes
mov eax, [rip + .Lend2]
.space 0x80000000
.Lend2:

## Backward: -0x80000000 fits, -0x80000001 overflows
## The mov instruction is 6 bytes, so .space 0x7ffffffa gives displacement -0x80000000.
.Lstart1:
.space 0x7ffffffa
mov eax, [rip + .Lstart1]

# CHECK: :[[#@LINE+3]]:17: error: value of -2147483649 is too large for field of 4 bytes
.Lstart2:
.space 0x7ffffffb
mov eax, [rip + .Lstart2]
