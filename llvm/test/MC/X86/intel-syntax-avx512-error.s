// RUN: not llvm-mc %s -triple x86_64-unknown-unknown -mcpu=knl -mattr=+avx512f -x86-asm-syntax=intel -output-asm-variant=1 -o /dev/null 2>&1 | FileCheck %s

// Validate that only OpMask/Zero mark may immediately follow destination
  vfmsub213ps zmm8{rn-sae} {k2}, zmm8, zmm8
// CHECK: error: Expected an op-mask register at this point
  vfmsub213ps zmm8{k2} {rn-sae}, zmm8, zmm8
// CHECK: error: Expected a {z} mark at this point
  vfmsub213ps zmm8{rn-sae}, zmm8, zmm8
// CHECK: error: Expected an op-mask register at this point
  vpcmpltd k5{k0}, zmm7, zmm24
// CHECK: error: Register k0 can't be used as write mask

