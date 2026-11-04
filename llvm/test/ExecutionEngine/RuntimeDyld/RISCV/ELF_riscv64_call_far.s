# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc --triple=riscv64 --filetype=obj -o %t/reloc.o %s
# RUN: llvm-rtdyld --triple=riscv64 --verify --check=%s %t/reloc.o \
# RUN:     --dummy-extern far_func=0x0123456789abcdef

## Regression test for the riscv64 out-of-range R_RISCV_CALL/R_RISCV_CALL_PLT
## literal-pool call stub (mirrors the AArch64/LoongArch64 stub mechanism).
##
## call_near's target is in range: it must be patched directly via the
## normal auipc+jalr encoding, with no stub involved (fast path unchanged).
##
## call_far's target (--dummy-extern far_func, ~8.2e16 bytes away) is far
## out of the auipc/jalr's +-2GB range: it must be redirected through a
## 24-byte literal-pool stub placed immediately after call_far, consisting
## of `auipc t0,0`, `ld t0,16(t0)`, `jalr x0,0(t0)`, `nop`, and an 8-byte
## absolute-address literal equal to far_func's address.

    .text
    .globl local_func
    .p2align 2
local_func:
    ret
    .size local_func, .-local_func

    .globl call_near
    .p2align 2
call_near:
    call local_func
    .size call_near, .-call_near

    .globl call_far
    .p2align 2
call_far:
    call far_func
    .size call_far, .-call_far

## call_near: direct patch, no stub -- auipc ra,0 / jalr ra,-4(ra)
# rtdyld-check: *{4}(call_near) = 0x00000097
# rtdyld-check: *{4}(call_near+4) = 0xffc080e7

## call_far: redirected to the stub 12 bytes ahead -- auipc ra,0 / jalr ra,12(ra)
# rtdyld-check: *{4}(call_far) = 0x00000097
# rtdyld-check: *{4}(call_far+4) = 0x00c080e7

## Stub body at call_far+12: auipc t0,0 ; ld t0,16(t0) ; jalr x0,0(t0) ; nop
# rtdyld-check: *{4}(call_far+12) = 0x00000297
# rtdyld-check: *{4}(call_far+16) = 0x0102b283
# rtdyld-check: *{4}(call_far+20) = 0x00028067
# rtdyld-check: *{4}(call_far+24) = 0x00000013

## Stub literal at call_far+28: absolute address of far_func
# rtdyld-check: *{8}(call_far+28) = 0x0123456789abcdef

