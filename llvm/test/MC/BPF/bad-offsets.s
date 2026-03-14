# RUN: not llvm-mc -mcpu=v4 -triple bpfel < %s 2>&1 \
# RUN:   | grep 'error: operand is not an identifier or 16-bit signed integer' \
# RUN:   | count 2
# RUN: not llvm-mc -mcpu=v4 -triple bpfel < %s 2>&1 \
# RUN:   | grep 'error: operand is not a 16-bit signed integer' \
# RUN:   | count 25
        if r1 > r2 goto +70000
        if r1 > r2 goto -70000
        *(u64 *)(r1 + 70000) = 10
        *(u32 *)(r1 - 70000) = 10
        *(u16 *)(r1 - 70000) = 10
        *(u8  *)(r1 - 70000) = 10
        *(u64 *)(r1 + 70000) = r1
        *(u32 *)(r1 - 70000) = r1
        *(u16 *)(r1 - 70000) = r1
        *(u8  *)(r1 - 70000) = r1
        r1 = *(u64 *)(r1 + 70000)
        r1 = *(u32 *)(r1 - 70000)
        r1 = *(u16 *)(r1 - 70000)
        r1 = *(u8  *)(r1 - 70000)
        r1 = *(s32 *)(r1 + 70000)
        r1 = *(s16 *)(r1 - 70000)
        r1 = *(s8  *)(r1 - 70000)
        lock *(u32*)(r1 + 70000) += w2
        lock *(u32*)(r1 - 70000) &= w2
        lock *(u32*)(r1 - 70000) |= w2
        lock *(u32*)(r1 - 70000) ^= w2
        r0 = atomic_fetch_add((u64 *)(r1 + 70000), r0)
        r0 = atomic_fetch_and((u64 *)(r1 + 70000), r0)
        r0 = atomic_fetch_xor((u64 *)(r1 + 70000), r0)
        r0 = atomic_fetch_or((u64 *)(r1 + 70000), r0)
        w0 = cmpxchg32_32(r1 + 70000, w0, w1)
        r0 = cmpxchg_64(r1 + 70000, r0, r1)
