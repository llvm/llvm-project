// RUN: not llvm-mc -triple x86_64-unknown-unknown --show-encoding %s 2>&1 | FileCheck %s

// CHECK: error: can't encode 'dh' in an instruction requiring REX prefix
movzx %dh, %rsi

// CHECK: error: can't encode 'ah' in an instruction requiring REX prefix
movzx %ah, %r8d

// CHECK: error: can't encode 'bh' in an instruction requiring REX prefix
add %bh, %sil

// CHECK: error: can't encode 'ch' in an instruction requiring REX prefix
mov %ch, (%r8)

// CHECK: error: can't encode 'dh' in an instruction requiring REX prefix
mov %dh, (%rax,%r8)
