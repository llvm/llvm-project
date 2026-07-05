# RUN: not llvm-mc -mcpu=v4 -triple bpfel < %s 2>&1 \
# RUN:   | grep 'error: operand is not the same as the dst register' \
# RUN:   | count 9
        r0 = bswap16 r1
        r0 = bswap32 r1
        r0 = bswap64 r1
        r0 = atomic_fetch_add((u64*)(r2 + 0), r1)
        r0 = atomic_fetch_and((u64*)(r2 + 0), r1)
        r0 = atomic_fetch_or((u64*)(r2 + 0), r1)
        r0 = atomic_fetch_xor((u64*)(r2 + 0), r1)
        w0 = xchg32_32(r2 + 0, w1)
        r0 = xchg_64(r2 + 0, r1)
