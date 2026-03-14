# RUN: not llvm-mc -triple x86_64 -show-encoding %s 2>&1 | FileCheck %s

# CHECK: [[#@LINE+2]]:1: error: instruction length exceeds the limit of 15
# CHECK: addq    $1234, %cs:-96, %rax
addq    $1234, %cs:-96, %rax

# CHECK: [[#@LINE+2]]:1: error: instruction length exceeds the limit of 15
# CHECK: subq    $1234, %fs:257(%rbx, %rcx), %rax
subq    $1234, %fs:257(%rbx, %rcx), %rax

# CHECK: [[#@LINE+2]]:1: error: instruction length exceeds the limit of 15
# CHECK: orq     $1234, 257(%ebx, %ecx), %rax
orq     $1234, 257(%ebx, %ecx), %rax

# CHECK: [[#@LINE+2]]:1: error: instruction length exceeds the limit of 15
# CHECK: xorq    $1234, %gs:257(%ebx), %rax
xorq    $1234, %gs:257(%ebx), %rax

# CHECK: [[#@LINE+2]]:1: error: instruction length exceeds the limit of 15
# CHECK: {nf} andq    $1234, %cs:-96
{nf} andq    $1234, %cs:-96

# CHECK: [[#@LINE+2]]:1: error: instruction length exceeds the limit of 15
# CHECK: {evex} adcq    $1234, %cs:-96
{evex} adcq    $1234, %cs:-96
