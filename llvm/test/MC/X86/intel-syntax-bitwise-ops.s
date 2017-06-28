// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=att %s | FileCheck %s

.intel_syntax

// CHECK: andl	$3, %ecx
    and ecx, 1+2
// CHECK: andl	$3, %ecx
    and ecx, 1|2
// CHECK: andl $3, %ecx
    and ecx, 1 or 2
// CHECK: andl $3, %ecx
    and ecx, 1 OR 2
// CHECK: andl $3, %ecx
    and ecx, 1*3
// CHECK: andl	$1, %ecx
    and ecx, 1&3
// CHECK: andl $1, %ecx
    and ecx, 1 and 3
// CHECK: andl $1, %ecx
    and ecx, 1 AND 3
// CHECK: andl $0, %ecx
    and ecx, (1&2)
// CHECK: andl $0, %ecx
    and ecx, (1 and 2)
// CHECK: andl $0, %ecx
    and ecx, (1 AND 2)
// CHECK: andl $3, %ecx
    and ecx, ((1)|2)
// CHECK: andl $3, %ecx
    and ecx, ((1) or 2)
// CHECK: andl $3, %ecx
    and ecx, ((1) OR 2)
// CHECK: andl $1, %ecx
    and ecx, 1&2+3
// CHECK: andl $1, %ecx
    and ecx, 1 and 2+3
// CHECK: andl $1, %ecx
    and ecx, 1 AND 2+3
// CHECK: addl $4938, %eax
    add eax, 9876 >> 1
// CHECK: addl $4938, %eax
    add eax, 9876 shr 1
// CHECK: addl $4938, %eax
    add eax, 9876 SHR 1
// CHECK: addl $19752, %eax
    add eax, 9876 << 1
// CHECK: addl $19752, %eax
    add eax, 9876 shl 1
// CHECK: addl $19752, %eax
    add eax, 9876 SHL 1
// CHECK: addl $5, %eax
    add eax, 6 ^ 3
// CHECK: addl $5, %eax
    add eax, 6 xor 3
// CHECK: addl $5, %eax
    add eax, 6 XOR 3
// CHECK: addl $5, %eax
    add eax, 6 XOR 3 shl 1 SHR 1
// CHECK: movl  $-9, %eax
    mov eax, not(1 shl 3)
// CHECK: movl    $-2, %eax
    mov eax, ~(0x8 shr 3)
// CHECK: movl  $-4, %eax
    mov eax,  not(1 or 3)
// CHECK: movl  $-2, %eax
    mov eax, -(1 xor 3)
// CHECK: movl  $-2, %eax
    mov eax, not(1 and 3)
// CHECK: movl  $3, %eax
    mov eax, not(not 3)
// CHECK: movl  $-3, %eax
    mov eax, ~(5 mod 3)
// CHECK: movl  $-2, %eax
    mov eax, (-5 mod 3)

