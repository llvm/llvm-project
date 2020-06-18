// RUN: not llvm-mc %s -triple x86_64-unknown-unknown --show-encoding -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s

// ERR: Register k0 can't be used as write mask
vpcmpd $1, %zmm24, %zmm7, %k5{%k0}

// ERR: Expected a {z} mark at this point
vfmsub213ps %zmm8, %zmm8, %zmm8{%k2} {rn-sae}

// ERR: Expected an op-mask register at this point
vfmsub213ps %zmm8, %zmm8, %zmm8 {rn-sae}

// ERR: invalid operand for instruction
cvtsd2sil  {rn-sae}, %xmm1, %eax

// ERR: Expected an identifier after {
cvtsd2sil  {{sae}, %xmm1, %eax

// ERR: invalid instruction mnemonic 'vpmuld'
vpmuld %xmm1, %xmm2, %xmm3

// ERR: invalid instruction mnemonic 'maskmov'
maskmov %mm1, %mm2
