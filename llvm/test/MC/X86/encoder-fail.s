// RUN: not llvm-mc -triple x86_64-unknown-unknown --show-encoding %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple x86_64-unknown-unknown --show-encoding -x86-asm-syntax=intel %s 2>&1 | FileCheck %s --check-prefix=CHECK-INTEL

// CHECK: error: can't encode 'dh' in an instruction requiring EVEX/REX2/REX prefix
movzx %dh, %rsi

// CHECK: error: can't encode 'ah' in an instruction requiring EVEX/REX2/REX prefix
movzx %ah, %r8d

// CHECK: error: can't encode 'bh' in an instruction requiring EVEX/REX2/REX prefix
add %bh, %sil

// CHECK: error: can't encode 'ch' in an instruction requiring EVEX/REX2/REX prefix
mov %ch, (%r8)

// CHECK: error: can't encode 'dh' in an instruction requiring EVEX/REX2/REX prefix
mov %dh, (%rax,%r8)

// CHECK-INTEL: error: can't encode 'ah' in an instruction requiring EVEX/REX2/REX prefix
add ah, ah, ah

// CHECK-INTEL: error: can't encode 'ah' in an instruction requiring EVEX/REX2/REX prefix
and ah, byte ptr [-13426159], ah

// CHECK-INTEL: error: can't encode 'ah' in an instruction requiring EVEX/REX2/REX prefix
ccmpa {dfv=of,cf} byte ptr [r8 + 4*rax + 291], ah

// CHECK-INTEL: error: can't encode 'ah' in an instruction requiring EVEX/REX2/REX prefix
ccmpae {dfv=of,cf} byte ptr [r8 + 4*rax + 291], ah

// CHECK-INTEL: error: can't encode 'ah' in an instruction requiring EVEX/REX2/REX prefix
sar ah, byte ptr [-13426159]

// CHECK-INTEL: error: can't encode 'ah' in an instruction requiring EVEX/REX2/REX prefix
{rex2} add ah, al

// CHECK-INTEL: error: can't encode 'ah' in an instruction requiring EVEX/REX2/REX prefix
{rex} add ah, al
