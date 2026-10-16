// RUN: llvm-mc -triple bpfel --mcpu=v4 --assemble --filetype=obj %s \
// RUN:   | llvm-objdump -d --mattr=+alu32 - \
// RUN:   | FileCheck %s

// CHECK: d7 01 00 00 10 00 00 00	r1 = bswap16 r1
// CHECK: d7 02 00 00 20 00 00 00	r2 = bswap32 r2
// CHECK: d7 03 00 00 40 00 00 00	r3 = bswap64 r3
r1 = bswap16 r1
r2 = bswap32 r2
r3 = bswap64 r3

// CHECK: 91 41 00 00 00 00 00 00	r1 = *(s8 *)(r4 + 0x0)
// CHECK: 89 52 04 00 00 00 00 00	r2 = *(s16 *)(r5 + 0x4)
// CHECK: 81 63 08 00 00 00 00 00	r3 = *(s32 *)(r6 + 0x8)
r1 = *(s8 *)(r4 + 0)
r2 = *(s16 *)(r5 + 4)
r3 = *(s32 *)(r6 + 8)

// CHECK: 91 41 00 00 00 00 00 00	r1 = *(s8 *)(r4 + 0x0)
// CHECK: 89 52 04 00 00 00 00 00	r2 = *(s16 *)(r5 + 0x4)
r1 = *(s8 *)(r4 + 0)
r2 = *(s16 *)(r5 + 4)

// CHECK: bf 41 08 00 00 00 00 00	r1 = (s8)r4
// CHECK: bf 52 10 00 00 00 00 00	r2 = (s16)r5
// CHECK: bf 63 20 00 00 00 00 00	r3 = (s32)r6
r1 = (s8)r4
r2 = (s16)r5
r3 = (s32)r6

// CHECK: bc 31 08 00 00 00 00 00	w1 = (s8)w3
// CHECK: bc 42 10 00 00 00 00 00	w2 = (s16)w4
w1 = (s8)w3
w2 = (s16)w4

// CHECK: 3f 31 01 00 00 00 00 00     r1 s/= r3
// CHECK: 9f 42 01 00 00 00 00 00     r2 s%= r4
r1 s/= r3
r2 s%= r4

// CHECK: 3c 31 01 00 00 00 00 00     w1 s/= w3
// CHECK: 9c 42 01 00 00 00 00 00     w2 s%= w4
w1 s/= w3
w2 s%= w4

// CHECK: d3 10 00 00 00 01 00 00	w0 = load_acquire((u8 *)(r1 + 0x0))
// CHECK: cb 10 00 00 00 01 00 00	w0 = load_acquire((u16 *)(r1 + 0x0))
// CHECK: c3 10 00 00 00 01 00 00	w0 = load_acquire((u32 *)(r1 + 0x0))
w0 = load_acquire((u8 *)(r1 + 0))
w0 = load_acquire((u16 *)(r1 + 0))
w0 = load_acquire((u32 *)(r1 + 0))

// CHECK: db 10 00 00 00 01 00 00	r0 = load_acquire((u64 *)(r1 + 0x0))
r0 = load_acquire((u64 *)(r1 + 0))

// CHECK: d3 21 00 00 10 01 00 00	store_release((u8 *)(r1 + 0x0), w2)
// CHECK: cb 21 00 00 10 01 00 00	store_release((u16 *)(r1 + 0x0), w2)
// CHECK: c3 21 00 00 10 01 00 00	store_release((u32 *)(r1 + 0x0), w2)
store_release((u8 *)(r1 + 0), w2)
store_release((u16 *)(r1 + 0), w2)
store_release((u32 *)(r1 + 0), w2)

// CHECK: db 21 00 00 10 01 00 00	store_release((u64 *)(r1 + 0x0), r2)
store_release((u64 *)(r1 + 0), r2)
