// RUN: llvm-mc -triple bpfel --mcpu=v3 --assemble --filetype=obj %s \
// RUN:   | llvm-objdump -d --mattr=+alu32 - \
// RUN:   | FileCheck %s

// CHECK: 07 01 00 00 2a 00 00 00	r1 += 0x2a
// CHECK: 0f 21 00 00 00 00 00 00	r1 += r2
r1 += 42
r1 += r2
// CHECK: 17 01 00 00 2a 00 00 00	r1 -= 0x2a
// CHECK: 1f 21 00 00 00 00 00 00	r1 -= r2
r1 -= 42
r1 -= r2
// CHECK: 27 01 00 00 2a 00 00 00	r1 *= 0x2a
// CHECK: 2f 21 00 00 00 00 00 00	r1 *= r2
r1 *= 42
r1 *= r2
// CHECK: 37 01 00 00 2a 00 00 00	r1 /= 0x2a
// CHECK: 3f 21 00 00 00 00 00 00	r1 /= r2
r1 /= 42
r1 /= r2
// CHECK: 47 01 00 00 2a 00 00 00	r1 |= 0x2a
// CHECK: 4f 21 00 00 00 00 00 00	r1 |= r2
r1 |= 42
r1 |= r2
// CHECK: 57 01 00 00 2a 00 00 00	r1 &= 0x2a
// CHECK: 5f 21 00 00 00 00 00 00	r1 &= r2
r1 &= 42
r1 &= r2
// CHECK: 67 01 00 00 2a 00 00 00	r1 <<= 0x2a
// CHECK: 6f 21 00 00 00 00 00 00	r1 <<= r2
r1 <<= 42
r1 <<= r2
// CHECK: 77 01 00 00 2a 00 00 00	r1 >>= 0x2a
// CHECK: 7f 21 00 00 00 00 00 00	r1 >>= r2
r1 >>= 42
r1 >>= r2
// CHECK: 97 01 00 00 2a 00 00 00	r1 %= 0x2a
// CHECK: 9f 21 00 00 00 00 00 00	r1 %= r2
r1 %= 42
r1 %= r2
// CHECK: a7 01 00 00 2a 00 00 00	r1 ^= 0x2a
// CHECK: af 21 00 00 00 00 00 00	r1 ^= r2
r1 ^= 42
r1 ^= r2
// CHECK: b7 01 00 00 2a 00 00 00	r1 = 0x2a
// CHECK: bf 21 00 00 00 00 00 00	r1 = r2
r1 = 42
r1 = r2
// CHECK: c7 01 00 00 2a 00 00 00	r1 s>>= 0x2a
// CHECK: cf 21 00 00 00 00 00 00	r1 s>>= r2
r1 s>>= 42
r1 s>>= r2
// CHECK: 87 01 00 00 00 00 00 00	r1 = -r1
r1 = -r1

// CHECK: 18 01 00 00 2a 00 00 00 00 00 00 00 00 00 00 00	r1 = 0x2a ll
r1 = 42 ll

// CHECK: 04 01 00 00 2a 00 00 00	w1 += 0x2a
// CHECK: 0c 21 00 00 00 00 00 00	w1 += w2
w1 += 42
w1 += w2
// CHECK: 14 01 00 00 2a 00 00 00	w1 -= 0x2a
// CHECK: 1c 21 00 00 00 00 00 00	w1 -= w2
w1 -= 42
w1 -= w2
// CHECK: 24 01 00 00 2a 00 00 00	w1 *= 0x2a
// CHECK: 2c 21 00 00 00 00 00 00	w1 *= w2
w1 *= 42
w1 *= w2
// CHECK: 34 01 00 00 2a 00 00 00	w1 /= 0x2a
// CHECK: 3c 21 00 00 00 00 00 00	w1 /= w2
w1 /= 42
w1 /= w2
// CHECK: 44 01 00 00 2a 00 00 00	w1 |= 0x2a
// CHECK: 4c 21 00 00 00 00 00 00	w1 |= w2
w1 |= 42
w1 |= w2
// CHECK: 54 01 00 00 2a 00 00 00	w1 &= 0x2a
// CHECK: 5c 21 00 00 00 00 00 00	w1 &= w2
w1 &= 42
w1 &= w2
// CHECK: 64 01 00 00 2a 00 00 00	w1 <<= 0x2a
// CHECK: 6c 21 00 00 00 00 00 00	w1 <<= w2
w1 <<= 42
w1 <<= w2
// CHECK: 74 01 00 00 2a 00 00 00	w1 >>= 0x2a
// CHECK: 7c 21 00 00 00 00 00 00	w1 >>= w2
w1 >>= 42
w1 >>= w2
// CHECK: 94 01 00 00 2a 00 00 00	w1 %= 0x2a
// CHECK: 9c 21 00 00 00 00 00 00	w1 %= w2
w1 %= 42
w1 %= w2
// CHECK: a4 01 00 00 2a 00 00 00	w1 ^= 0x2a
// CHECK: ac 21 00 00 00 00 00 00	w1 ^= w2
w1 ^= 42
w1 ^= w2
// CHECK: b4 01 00 00 2a 00 00 00	w1 = 0x2a
// CHECK: bc 21 00 00 00 00 00 00	w1 = w2
w1 = 42
w1 = w2
// CHECK: c4 01 00 00 2a 00 00 00	w1 s>>= 0x2a
// CHECK: cc 21 00 00 00 00 00 00	w1 s>>= w2
w1 s>>= 42
w1 s>>= w2
// CHECK: 84 01 00 00 00 00 00 00	w1 = -w1
w1 = -w1

// CHECK: dc 01 00 00 10 00 00 00	r1 = be16 r1
// CHECK: dc 02 00 00 20 00 00 00	r2 = be32 r2
// CHECK: dc 03 00 00 40 00 00 00	r3 = be64 r3
r1 = be16 r1
r2 = be32 r2
r3 = be64 r3
// CHECK: d4 01 00 00 10 00 00 00	r1 = le16 r1
// CHECK: d4 02 00 00 20 00 00 00	r2 = le32 r2
// CHECK: d4 03 00 00 40 00 00 00	r3 = le64 r3
r1 = le16 r1
r2 = le32 r2
r3 = le64 r3
// CHECK: 05 00 00 00 00 00 00 00	goto +0x0
goto +0

// CHECK: 15 01 00 00 2a 00 00 00	if r1 == 0x2a goto +0x0
// CHECK: 1d 21 00 00 00 00 00 00	if r1 == r2 goto +0x0
if r1 == 42 goto +0
if r1 == r2 goto +0
// CHECK: 55 01 00 00 2a 00 00 00	if r1 != 0x2a goto +0x0
// CHECK: 5d 21 00 00 00 00 00 00	if r1 != r2 goto +0x0
if r1 != 42 goto +0
if r1 != r2 goto +0
// CHECK: 25 01 00 00 2a 00 00 00	if r1 > 0x2a goto +0x0
// CHECK: 2d 21 00 00 00 00 00 00	if r1 > r2 goto +0x0
if r1 > 42 goto +0
if r1 > r2 goto +0
// CHECK: 35 01 00 00 2a 00 00 00	if r1 >= 0x2a goto +0x0
// CHECK: 3d 21 00 00 00 00 00 00	if r1 >= r2 goto +0x0
if r1 >= 42 goto +0
if r1 >= r2 goto +0
// CHECK: 65 01 00 00 2a 00 00 00	if r1 s> 0x2a goto +0x0
// CHECK: 6d 21 00 00 00 00 00 00	if r1 s> r2 goto +0x0
if r1 s> 42 goto +0
if r1 s> r2 goto +0
// CHECK: 75 01 00 00 2a 00 00 00	if r1 s>= 0x2a goto +0x0
// CHECK: 7d 21 00 00 00 00 00 00	if r1 s>= r2 goto +0x0
if r1 s>= 42 goto +0
if r1 s>= r2 goto +0
// CHECK: a5 01 00 00 2a 00 00 00	if r1 < 0x2a goto +0x0
// CHECK: ad 21 00 00 00 00 00 00	if r1 < r2 goto +0x0
if r1 < 42 goto +0
if r1 < r2 goto +0
// CHECK: b5 01 00 00 2a 00 00 00	if r1 <= 0x2a goto +0x0
// CHECK: bd 21 00 00 00 00 00 00	if r1 <= r2 goto +0x0
if r1 <= 42 goto +0
if r1 <= r2 goto +0
// CHECK: c5 01 00 00 2a 00 00 00	if r1 s< 0x2a goto +0x0
// CHECK: cd 21 00 00 00 00 00 00	if r1 s< r2 goto +0x0
if r1 s< 42 goto +0
if r1 s< r2 goto +0
// CHECK: d5 01 00 00 2a 00 00 00	if r1 s<= 0x2a goto +0x0
// CHECK: dd 21 00 00 00 00 00 00	if r1 s<= r2 goto +0x0
if r1 s<= 42 goto +0
if r1 s<= r2 goto +0

// CHECK: 16 01 00 00 2a 00 00 00	if w1 == 0x2a goto +0x0
// CHECK: 1e 21 00 00 00 00 00 00	if w1 == w2 goto +0x0
if w1 == 42 goto +0
if w1 == w2 goto +0
// CHECK: 56 01 00 00 2a 00 00 00	if w1 != 0x2a goto +0x0
// CHECK: 5e 21 00 00 00 00 00 00	if w1 != w2 goto +0x0
if w1 != 42 goto +0
if w1 != w2 goto +0
// CHECK: 26 01 00 00 2a 00 00 00	if w1 > 0x2a goto +0x0
// CHECK: 2e 21 00 00 00 00 00 00	if w1 > w2 goto +0x0
if w1 > 42 goto +0
if w1 > w2 goto +0
// CHECK: 36 01 00 00 2a 00 00 00	if w1 >= 0x2a goto +0x0
// CHECK: 3e 21 00 00 00 00 00 00	if w1 >= w2 goto +0x0
if w1 >= 42 goto +0
if w1 >= w2 goto +0
// CHECK: 66 01 00 00 2a 00 00 00	if w1 s> 0x2a goto +0x0
// CHECK: 6e 21 00 00 00 00 00 00	if w1 s> w2 goto +0x0
if w1 s> 42 goto +0
if w1 s> w2 goto +0
// CHECK: 76 01 00 00 2a 00 00 00	if w1 s>= 0x2a goto +0x0
// CHECK: 7e 21 00 00 00 00 00 00	if w1 s>= w2 goto +0x0
if w1 s>= 42 goto +0
if w1 s>= w2 goto +0
// CHECK: a6 01 00 00 2a 00 00 00	if w1 < 0x2a goto +0x0
// CHECK: ae 21 00 00 00 00 00 00	if w1 < w2 goto +0x0
if w1 < 42 goto +0
if w1 < w2 goto +0
// CHECK: b6 01 00 00 2a 00 00 00	if w1 <= 0x2a goto +0x0
// CHECK: be 21 00 00 00 00 00 00	if w1 <= w2 goto +0x0
if w1 <= 42 goto +0
if w1 <= w2 goto +0
// CHECK: c6 01 00 00 2a 00 00 00	if w1 s< 0x2a goto +0x0
// CHECK: ce 21 00 00 00 00 00 00	if w1 s< w2 goto +0x0
if w1 s< 42 goto +0
if w1 s< w2 goto +0
// CHECK: d6 01 00 00 2a 00 00 00	if w1 s<= 0x2a goto +0x0
// CHECK: de 21 00 00 00 00 00 00	if w1 s<= w2 goto +0x0
if w1 s<= 42 goto +0
if w1 s<= w2 goto +0

// CHECK: 85 00 00 00 2a 00 00 00	call 0x2a
call +42
// CHECK: 95 00 00 00 00 00 00 00	exit
exit

// Note: For the group below w1 is used as a destination for sizes u8, u16, u32.
//       This is disassembler quirk, but is technically not wrong, as there are
//       no different encodings for 'r1 = load' vs 'w1 = load'.
//
// CHECK: 71 21 2a 00 00 00 00 00	w1 = *(u8 *)(r2 + 0x2a)
// CHECK: 69 21 2a 00 00 00 00 00	w1 = *(u16 *)(r2 + 0x2a)
// CHECK: 61 21 2a 00 00 00 00 00	w1 = *(u32 *)(r2 + 0x2a)
// CHECK: 79 21 2a 00 00 00 00 00	r1 = *(u64 *)(r2 + 0x2a)
r1 = *(u8*)(r2 + 42)
r1 = *(u16*)(r2 + 42)
r1 = *(u32*)(r2 + 42)
r1 = *(u64*)(r2 + 42)

// Note: For the group below w1 is used as a source for sizes u8, u16, u32.
//       This is disassembler quirk, but is technically not wrong, as there are
//       no different encodings for 'store r1' vs 'store w1'.
//
// CHECK: 73 12 2a 00 00 00 00 00	*(u8 *)(r2 + 0x2a) = w1
// CHECK: 6b 12 2a 00 00 00 00 00	*(u16 *)(r2 + 0x2a) = w1
// CHECK: 63 12 2a 00 00 00 00 00	*(u32 *)(r2 + 0x2a) = w1
// CHECK: 7b 12 2a 00 00 00 00 00	*(u64 *)(r2 + 0x2a) = r1
*(u8*)(r2 + 42) = r1
*(u16*)(r2 + 42) = r1
*(u32*)(r2 + 42) = r1
*(u64*)(r2 + 42) = r1

// CHECK: c3 21 01 00 00 00 00 00	lock *(u32 *)(r1 + 0x1) += w2
// CHECK: c3 21 01 00 50 00 00 00	lock *(u32 *)(r1 + 0x1) &= w2
// CHECK: c3 21 01 00 40 00 00 00	lock *(u32 *)(r1 + 0x1) |= w2
// CHECK: c3 21 01 00 a0 00 00 00	lock *(u32 *)(r1 + 0x1) ^= w2
lock *(u32*)(r1 + 1) += w2
lock *(u32*)(r1 + 1) &= w2
lock *(u32*)(r1 + 1) |= w2
lock *(u32*)(r1 + 1) ^= w2
// CHECK: db 21 01 00 00 00 00 00	lock *(u64 *)(r1 + 0x1) += r2
// CHECK: db 21 01 00 50 00 00 00	lock *(u64 *)(r1 + 0x1) &= r2
// CHECK: db 21 01 00 40 00 00 00	lock *(u64 *)(r1 + 0x1) |= r2
// CHECK: db 21 01 00 a0 00 00 00	lock *(u64 *)(r1 + 0x1) ^= r2
lock *(u64*)(r1 + 1) += r2
lock *(u64*)(r1 + 1) &= r2
lock *(u64*)(r1 + 1) |= r2
lock *(u64*)(r1 + 1) ^= r2
// CHECK: c3 01 00 00 01 00 00 00	w0 = atomic_fetch_add((u32 *)(r1 + 0x0), w0)
// CHECK: c3 01 00 00 51 00 00 00	w0 = atomic_fetch_and((u32 *)(r1 + 0x0), w0)
// CHECK: c3 01 00 00 a1 00 00 00	w0 = atomic_fetch_xor((u32 *)(r1 + 0x0), w0)
// CHECK: c3 01 00 00 41 00 00 00	w0 = atomic_fetch_or((u32 *)(r1 + 0x0), w0)
w0 = atomic_fetch_add((u32 *)(r1 + 0), w0)
w0 = atomic_fetch_and((u32 *)(r1 + 0), w0)
w0 = atomic_fetch_xor((u32 *)(r1 + 0), w0)
w0 = atomic_fetch_or((u32 *)(r1 + 0), w0)
// CHECK: db 01 00 00 01 00 00 00	r0 = atomic_fetch_add((u64 *)(r1 + 0x0), r0)
// CHECK: db 01 00 00 51 00 00 00	r0 = atomic_fetch_and((u64 *)(r1 + 0x0), r0)
// CHECK: db 01 00 00 a1 00 00 00	r0 = atomic_fetch_xor((u64 *)(r1 + 0x0), r0)
// CHECK: db 01 00 00 41 00 00 00	r0 = atomic_fetch_or((u64 *)(r1 + 0x0), r0)
r0 = atomic_fetch_add((u64 *)(r1 + 0), r0)
r0 = atomic_fetch_and((u64 *)(r1 + 0), r0)
r0 = atomic_fetch_xor((u64 *)(r1 + 0), r0)
r0 = atomic_fetch_or((u64 *)(r1 + 0), r0)
// CHECK: c3 01 00 00 e1 00 00 00	w0 = xchg32_32(r1 + 0x0, w0)
// CHECK: db 01 00 00 e1 00 00 00	r0 = xchg_64(r1 + 0x0, r0)
w0 = xchg32_32(r1 + 0, w0)
r0 = xchg_64(r1 + 0, r0)
// CHECK: c3 11 00 00 f1 00 00 00	w0 = cmpxchg32_32(r1 + 0x0, w0, w1)
// CHECK: db 11 00 00 f1 00 00 00	r0 = cmpxchg_64(r1 + 0x0, r0, r1)
w0 = cmpxchg32_32(r1 + 0, w0, w1)
r0 = cmpxchg_64(r1 + 0, r0, r1)

// CHECK: 30 00 00 00 2a 00 00 00	r0 = *(u8 *)skb[0x2a]
// CHECK: 28 00 00 00 2a 00 00 00	r0 = *(u16 *)skb[0x2a]
// CHECK: 20 00 00 00 2a 00 00 00	r0 = *(u32 *)skb[0x2a]
r0 = *(u8*)skb[42]
r0 = *(u16*)skb[42]
r0 = *(u32*)skb[42]

// CHECK: 50 10 00 00 00 00 00 00	r0 = *(u8 *)skb[r1]
// CHECK: 48 10 00 00 00 00 00 00	r0 = *(u16 *)skb[r1]
// CHECK: 40 10 00 00 00 00 00 00	r0 = *(u32 *)skb[r1]
r0 = *(u8*)skb[r1]
r0 = *(u16*)skb[r1]
r0 = *(u32*)skb[r1]
