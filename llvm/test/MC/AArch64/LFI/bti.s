// RUN: llvm-mc -triple aarch64_lfi -mattr=+bti %s | FileCheck %s

// LFI routes indirect tail calls through x28 instead of x16/x17 so BTI C
// landing pads must be promoted to BTI JC.

bti c
// CHECK: bti jc

// Other BTI variants are left unchanged.
bti
// CHECK: bti{{$}}

bti j
// CHECK: bti j{{$}}

bti jc
// CHECK: bti jc
