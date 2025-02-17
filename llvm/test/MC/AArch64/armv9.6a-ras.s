// RUN: not llvm-mc -triple aarch64 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix NO-POPS
// RUN: llvm-mc -triple aarch64 -mattr=+pops -show-encoding < %s 2>&1 | FileCheck %s --check-prefix HAS-POPS

dc CIGDVAPS, x3
dc CIVAPS, x3
// NO-POPS: error: DC CIGDVAPS requires: pops
// NO-POPS: error: DC CIVAPS requires: pops

# HAS-POPS:      	dc	cigdvaps, x3                    // encoding: [0xa3,0x7f,0x08,0xd5]
# HAS-POPS-NEXT: 	dc	civaps, x3                      // encoding: [0x23,0x7f,0x08,0xd5]
