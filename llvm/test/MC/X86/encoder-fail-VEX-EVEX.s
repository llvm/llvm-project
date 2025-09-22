// RUN: not llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel --show-encoding %s 2>&1 | FileCheck %s

// CHECK: error: can't encode 'ah' in a VEX/EVEX-prefixed instruction
add ah, ah, ah

// CHECK: error: can't encode 'ah' in a VEX/EVEX-prefixed instruction
and ah, byte ptr [-13426159], ah

// CHECK: error: can't encode 'ah' in a VEX/EVEX-prefixed instruction
ccmpa {dfv=of,cf} byte ptr [r8 + 4*rax + 291], ah

// CHECK: error: can't encode 'ah' in a VEX/EVEX-prefixed instruction
ccmpae {dfv=of,cf} byte ptr [r8 + 4*rax + 291], ah

// CHECK: error: can't encode 'ah' in a VEX/EVEX-prefixed instruction
sar ah, byte ptr [-13426159]