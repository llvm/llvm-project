# RUN: llvm-mc -triple bpfel -filetype=obj -o %t %s
# RUN: llvm-objdump --no-print-imm-hex -d -r %t | FileCheck --check-prefixes CHECK,CHECK-64 %s
# RUN: llvm-objdump --no-print-imm-hex --mattr=+alu32 -d -r %t | FileCheck --check-prefixes CHECK,CHECK-32 %s

// ======== BPF_LD Class ========
// Some extra whitespaces are deliberately added to test the parser.

  r0 = * (u8 *)skb[0]    // BPF_LD | BPF_ABS | BPF_B
  r0 = *(u16 *)skb[2]    // BPF_LD | BPF_ABS | BPF_H
  r0 = * (u32*)skb[4]    // BPF_LD | BPF_ABS | BPF_W
// CHECK: 30 00 00 00 00 00 00 00 	r0 = *(u8 *)skb[0]
// CHECK: 28 00 00 00 02 00 00 00 	r0 = *(u16 *)skb[2]
// CHECK: 20 00 00 00 04 00 00 00 	r0 = *(u32 *)skb[4]

  r0 = * (u8 *)skb[r0]   // BPF_LD | BPF_IND | BPF_B
  r0 = *  (u16 *)skb[r1] // BPF_LD | BPF_IND | BPF_H
  r0 = *(u32 *)skb[r2]   // BPF_LD | BPF_IND | BPF_W
// CHECK: 50 00 00 00 00 00 00 00 	r0 = *(u8 *)skb[r0]
// CHECK: 48 10 00 00 00 00 00 00 	r0 = *(u16 *)skb[r1]
// CHECK: 40 20 00 00 00 00 00 00 	r0 = *(u32 *)skb[r2]

  r9 = 0xffffffff ll     // BPF_LD | BPF_DW | BPF_IMM
  r9 = 8589934591 ll     // BPF_LD | BPF_DW | BPF_IMM
  r9 = 0x1ffffffff ll    // BPF_LD | BPF_DW | BPF_IMM
  r9 = dummy_map  ll     // BPF_LD | BPF_DW | BPF_IMM
// CHECK: 18 09 00 00 ff ff ff ff 00 00 00 00 00 00 00 00 	r9 = 4294967295 ll
// CHECK: 18 09 00 00 ff ff ff ff 00 00 00 00 01 00 00 00 	r9 = 8589934591 ll
// CHECK: 18 09 00 00 ff ff ff ff 00 00 00 00 01 00 00 00 	r9 = 8589934591 ll
// CHECK: 18 09 00 00 00 00 00 00 00 00 00 00 00 00 00 00 	r9 = 0 ll
// CHECK: 0000000000000060:  R_BPF_64_64	dummy_map

// ======== BPF_LDX Class ========
  r5 = *(u8 *)(r0 + 0)   // BPF_LDX | BPF_B
  r6 = *(u16 *)(r1 + 8)  // BPF_LDX | BPF_H
  r7 = *(u32 *)(r2 + 16) // BPF_LDX | BPF_W
  r8 = *(u64 *)(r3 - 30) // BPF_LDX | BPF_DW
// CHECK-64: 71 05 00 00 00 00 00 00 	r5 = *(u8 *)(r0 + 0)
// CHECK-64: 69 16 08 00 00 00 00 00 	r6 = *(u16 *)(r1 + 8)
// CHECK-64: 61 27 10 00 00 00 00 00 	r7 = *(u32 *)(r2 + 16)
// CHECK-32: 71 05 00 00 00 00 00 00 	w5 = *(u8 *)(r0 + 0)
// CHECK-32: 69 16 08 00 00 00 00 00 	w6 = *(u16 *)(r1 + 8)
// CHECK-32: 61 27 10 00 00 00 00 00 	w7 = *(u32 *)(r2 + 16)
// CHECK: 79 38 e2 ff 00 00 00 00 	r8 = *(u64 *)(r3 - 30)

// ======== BPF_STX Class ========
  *(u8 *)(r0 + 0) = r7    // BPF_STX | BPF_B
  *(u16 *)(r1 + 8) = r8   // BPF_STX | BPF_H
  *(u32 *)(r2 + 16) = r9  // BPF_STX | BPF_W
  *(u64 *)(r3 - 30) = r10 // BPF_STX | BPF_DW
// CHECK-64: 73 70 00 00 00 00 00 00 	*(u8 *)(r0 + 0) = r7
// CHECK-64: 6b 81 08 00 00 00 00 00 	*(u16 *)(r1 + 8) = r8
// CHECK-64: 63 92 10 00 00 00 00 00 	*(u32 *)(r2 + 16) = r9
// CHECK-32: 73 70 00 00 00 00 00 00 	*(u8 *)(r0 + 0) = w7
// CHECK-32: 6b 81 08 00 00 00 00 00 	*(u16 *)(r1 + 8) = w8
// CHECK-32: 63 92 10 00 00 00 00 00 	*(u32 *)(r2 + 16) = w9
// CHECK: 7b a3 e2 ff 00 00 00 00 	*(u64 *)(r3 - 30) = r10

  lock *(u32 *)(r2 + 16) += r9  // BPF_STX | BPF_W | BPF_XADD
  lock *(u64 *)(r3 - 30) += r10 // BPF_STX | BPF_DW | BPF_XADD
// CHECK-64: c3 92 10 00 00 00 00 00 	lock *(u32 *)(r2 + 16) += r9
// CHECK-32: c3 92 10 00 00 00 00 00 	lock *(u32 *)(r2 + 16) += w9
// CHECK: db a3 e2 ff 00 00 00 00 	lock *(u64 *)(r3 - 30) += r10

// ======== BPF_JMP Class ========
  if r1 & r2 goto Llabel0    // BPF_JSET  | BPF_X
  if r1 & 0xffff goto Llabel0    // BPF_JSET  | BPF_K
// CHECK: 4d 21 1d 00 00 00 00 00 	if r1 & r2 goto +29
// CHECK: 45 01 1c 00 ff ff 00 00 	if r1 & 65535 goto +28

  goto Llabel0               // BPF_JA
  call 1                     // BPF_CALL
  exit                       // BPF_EXIT
// CHECK: 05 00 1b 00 00 00 00 00 	goto +27
// CHECK: 85 00 00 00 01 00 00 00 	call 1
// CHECK: 95 00 00 00 00 00 00 00 	exit

  if r0 == r1 goto Llabel0   // BPF_JEQ  | BPF_X
  if r3 != r4 goto Llabel0   // BPF_JNE  | BPF_X
// CHECK: 1d 10 18 00 00 00 00 00 	if r0 == r1 goto +24
// CHECK: 5d 43 17 00 00 00 00 00 	if r3 != r4 goto +23

  if r1 > r2 goto Llabel0    // BPF_JGT  | BPF_X
  if r2 >= r3 goto Llabel0   // BPF_JGE  | BPF_X
  if r4 s> r5 goto Llabel0   // BPF_JSGT | BPF_X
  if r5 s>= r6 goto Llabel0  // BPF_JSGE | BPF_X
// CHECK: 2d 21 16 00 00 00 00 00 	if r1 > r2 goto +22
// CHECK: 3d 32 15 00 00 00 00 00 	if r2 >= r3 goto +21
// CHECK: 6d 54 14 00 00 00 00 00 	if r4 s> r5 goto +20
// CHECK: 7d 65 13 00 00 00 00 00 	if r5 s>= r6 goto +19

  if r6 < r7 goto Llabel0    // BPF_JLT  | BPF_X
  if r7 <= r8 goto Llabel0   // BPF_JLE  | BPF_X
  if r8 s< r9 goto Llabel0   // BPF_JSLT | BPF_X
  if r9 s<= r10 goto Llabel0 // BPF_JSLE | BPF_X
// CHECK: ad 76 12 00 00 00 00 00 	if r6 < r7 goto +18
// CHECK: bd 87 11 00 00 00 00 00 	if r7 <= r8 goto +17
// CHECK: cd 98 10 00 00 00 00 00 	if r8 s< r9 goto +16
// CHECK: dd a9 0f 00 00 00 00 00 	if r9 s<= r10 goto +15

  if r0 == 0 goto Llabel0           // BPF_JEQ  | BPF_K
  if r3 != -1 goto Llabel0          // BPF_JNE  | BPF_K
// CHECK: 15 00 0e 00 00 00 00 00 	if r0 == 0 goto +14
// CHECK: 55 03 0d 00 ff ff ff ff 	if r3 != -1 goto +13

  if r1 > 64 goto Llabel0           // BPF_JGT  | BPF_K
  if r2 >= 0xffffffff goto Llabel0  // BPF_JGE  | BPF_K
  if r4 s> 0xffffffff goto Llabel0  // BPF_JSGT | BPF_K
  if r5 s>= 0x7fffffff goto Llabel0 // BPF_JSGE | BPF_K
// CHECK: 25 01 0c 00 40 00 00 00 	if r1 > 64 goto +12
// CHECK: 35 02 0b 00 ff ff ff ff 	if r2 >= -1 goto +11
// CHECK: 65 04 0a 00 ff ff ff ff 	if r4 s> -1 goto +10
// CHECK: 75 05 09 00 ff ff ff 7f 	if r5 s>= 2147483647 goto +9

  if r6 < 0xff goto Llabel0         // BPF_JLT  | BPF_K
  if r7 <= 0xffff goto Llabel0      // BPF_JLE  | BPF_K
  if r8 s< 0 goto Llabel0           // BPF_JSLT | BPF_K
  if r9 s<= -1 goto Llabel0         // BPF_JSLE | BPF_K
// CHECK: a5 06 08 00 ff 00 00 00 	if r6 < 255 goto +8
// CHECK: b5 07 07 00 ff ff 00 00 	if r7 <= 65535 goto +7
// CHECK: c5 08 06 00 00 00 00 00 	if r8 s< 0 goto +6
// CHECK: d5 09 05 00 ff ff ff ff 	if r9 s<= -1 goto +5

// ======== BPF_ALU64 Class ========
  r0 += r1    // BPF_ADD  | BPF_X
  r1 -= r2    // BPF_SUB  | BPF_X
  r2 *= r3    // BPF_MUL  | BPF_X
  r3 /= r4    // BPF_DIV  | BPF_X
  r4 %= r5    // BPF_MOD  | BPF_X
// CHECK: 0f 10 00 00 00 00 00 00 	r0 += r1
// CHECK: 1f 21 00 00 00 00 00 00 	r1 -= r2
// CHECK: 2f 32 00 00 00 00 00 00 	r2 *= r3
// CHECK: 3f 43 00 00 00 00 00 00 	r3 /= r4
// CHECK: 9f 54 00 00 00 00 00 00 	r4 %= r5

Llabel0 :
  r2 = -r2    // BPF_NEG
  r4 |= r5    // BPF_OR   | BPF_X
  r5 &= r6    // BPF_AND  | BPF_X
  r6 <<= r7   // BPF_LSH  | BPF_X
  r7 >>= r8   // BPF_RSH  | BPF_X
  r8 ^= r9    // BPF_XOR  | BPF_X
  r9 = r10    // BPF_MOV  | BPF_X
  r10 s>>= r0 // BPF_ARSH | BPF_X
// CHECK: <Llabel0>:
// CHECK: 87 02 00 00 00 00 00 00	r2 = -r2
// CHECK: 4f 54 00 00 00 00 00 00 	r4 |= r5
// CHECK: 5f 65 00 00 00 00 00 00 	r5 &= r6
// CHECK: 6f 76 00 00 00 00 00 00 	r6 <<= r7
// CHECK: 7f 87 00 00 00 00 00 00 	r7 >>= r8
// CHECK: af 98 00 00 00 00 00 00 	r8 ^= r9
// CHECK: bf a9 00 00 00 00 00 00 	r9 = r10
// CHECK: cf 0a 00 00 00 00 00 00 	r10 s>>= r0

  r1 = be16 r1  // BPF_END  | BPF_TO_BE
  r2 = be32 r2  // BPF_END  | BPF_TO_BE
  r3 = be64 r3  // BPF_END  | BPF_TO_BE
// CHECK: dc 01 00 00 10 00 00 00 	r1 = be16 r1
// CHECK: dc 02 00 00 20 00 00 00 	r2 = be32 r2
// CHECK: dc 03 00 00 40 00 00 00 	r3 = be64 r3

  r0 += 1           // BPF_ADD  | BPF_K
  r1 -= 0x1         // BPF_SUB  | BPF_K
  r2 *= -4          // BPF_MUL  | BPF_K
  r3 /= 5           // BPF_DIV  | BPF_K
  r4 %= 5           // BPF_MOD  | BPF_K
// CHECK: 07 00 00 00 01 00 00 00 	r0 += 1
// CHECK: 17 01 00 00 01 00 00 00 	r1 -= 1
// CHECK: 27 02 00 00 fc ff ff ff 	r2 *= -4
// CHECK: 37 03 00 00 05 00 00 00 	r3 /= 5
// CHECK: 97 04 00 00 05 00 00 00 	r4 %= 5

  r4 |= 0xff        // BPF_OR   | BPF_K
  r5 &= 0xFF        // BPF_AND  | BPF_K
  r6 <<= 63         // BPF_LSH  | BPF_K
  r7 >>= 32         // BPF_RSH  | BPF_K
  r8 ^= 0           // BPF_XOR  | BPF_K
  r9 = 1            // BPF_MOV  | BPF_K
  r9 = 0xffffffff   // BPF_MOV  | BPF_K
  r10 s>>= 64       // BPF_ARSH | BPF_K
// CHECK: 47 04 00 00 ff 00 00 00 	r4 |= 255
// CHECK: 57 05 00 00 ff 00 00 00 	r5 &= 255
// CHECK: 67 06 00 00 3f 00 00 00 	r6 <<= 63
// CHECK: 77 07 00 00 20 00 00 00 	r7 >>= 32
// CHECK: a7 08 00 00 00 00 00 00 	r8 ^= 0
// CHECK: b7 09 00 00 01 00 00 00 	r9 = 1
// CHECK: b7 09 00 00 ff ff ff ff 	r9 = -1
// CHECK: c7 0a 00 00 40 00 00 00 	r10 s>>= 64
