# RUN: llvm-mc -triple bpfel -filetype=obj -o %t %s
# RUN: llvm-objdump -d -r %t | FileCheck %s
# RUN: llvm-objdump -mattr=+alu32 -d -r %t | FileCheck %s

// ======== BPF_ALU Class ========
  w1 = -w1    // BPF_NEG
  w0 += w1    // BPF_ADD  | BPF_X
  w1 -= w2    // BPF_SUB  | BPF_X
  w2 *= w3    // BPF_MUL  | BPF_X
  w3 /= w4    // BPF_DIV  | BPF_X
// CHECK: 84 01 00 00 00 00 00 00      w1 = -w1
// CHECK: 0c 10 00 00 00 00 00 00      w0 += w1
// CHECK: 1c 21 00 00 00 00 00 00      w1 -= w2
// CHECK: 2c 32 00 00 00 00 00 00      w2 *= w3
// CHECK: 3c 43 00 00 00 00 00 00      w3 /= w4

  w4 |= w5    // BPF_OR   | BPF_X
  w5 &= w6    // BPF_AND  | BPF_X
  w6 <<= w7   // BPF_LSH  | BPF_X
  w7 >>= w8   // BPF_RSH  | BPF_X
  w8 ^= w9    // BPF_XOR  | BPF_X
  w9 = w10    // BPF_MOV  | BPF_X
  w10 s>>= w0 // BPF_ARSH | BPF_X
// CHECK: 4c 54 00 00 00 00 00 00      w4 |= w5
// CHECK: 5c 65 00 00 00 00 00 00      w5 &= w6
// CHECK: 6c 76 00 00 00 00 00 00      w6 <<= w7
// CHECK: 7c 87 00 00 00 00 00 00      w7 >>= w8
// CHECK: ac 98 00 00 00 00 00 00      w8 ^= w9
// CHECK: bc a9 00 00 00 00 00 00      w9 = w10
// CHECK: cc 0a 00 00 00 00 00 00      w10 s>>= w0

  w0 += 1           // BPF_ADD  | BPF_K
  w1 -= 0x1         // BPF_SUB  | BPF_K
  w2 *= -4          // BPF_MUL  | BPF_K
  w3 /= 5           // BPF_DIV  | BPF_K
// CHECK: 04 00 00 00 01 00 00 00      w0 += 1
// CHECK: 14 01 00 00 01 00 00 00      w1 -= 1
// CHECK: 24 02 00 00 fc ff ff ff      w2 *= -4
// CHECK: 34 03 00 00 05 00 00 00      w3 /= 5

  w4 |= 0xff        // BPF_OR   | BPF_K
  w5 &= 0xFF        // BPF_AND  | BPF_K
  w6 <<= 63         // BPF_LSH  | BPF_K
  w7 >>= 32         // BPF_RSH  | BPF_K
  w8 ^= 0           // BPF_XOR  | BPF_K
  w9 = 1            // BPF_MOV  | BPF_K
  w9 = 0xffffffff   // BPF_MOV  | BPF_K
  w10 s>>= 64       // BPF_ARSH | BPF_K
// CHECK: 44 04 00 00 ff 00 00 00      w4 |= 255
// CHECK: 54 05 00 00 ff 00 00 00      w5 &= 255
// CHECK: 64 06 00 00 3f 00 00 00      w6 <<= 63
// CHECK: 74 07 00 00 20 00 00 00      w7 >>= 32
// CHECK: a4 08 00 00 00 00 00 00      w8 ^= 0
// CHECK: b4 09 00 00 01 00 00 00      w9 = 1
// CHECK: b4 09 00 00 ff ff ff ff      w9 = -1
// CHECK: c4 0a 00 00 40 00 00 00      w10 s>>= 64
