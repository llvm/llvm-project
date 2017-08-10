// RUN: not llvm-mc -triple x86_64-unknown-unknown -mcpu=knl -mattr=+avx512dq -mattr=+avx512f --show-encoding %s 2> %t.err
// RUN: FileCheck --check-prefix=ERR < %t.err %s

// ERR: invalid operand for instruction
vpcmpd $1, %zmm24, %zmm7, %k5{%k0}

// ERR: Expected a {z} mark at this point
vfmsub213ps %zmm8, %zmm8, %zmm8{%k2} {rn-sae}

// ERR: Expected an op-mask register at this point
vfmsub213ps %zmm8, %zmm8, %zmm8 {rn-sae}
