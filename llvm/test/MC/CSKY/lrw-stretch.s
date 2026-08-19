# RUN: llvm-mc -triple=csky -mattr=+e2 -filetype=obj -o %t %s
# RUN: llvm-objdump --mattr=+e2 --no-show-raw-insn -M no-aliases -d %t | FileCheck %s

## The two "br16 external" relax from 2-byte to 4-byte (unresolved symbol).
## During the fused relaxation+layout pass, the cumulative Stretch must not
## cause the lrw16 instructions to appear misaligned and spuriously relax to
## 4-byte lrw32. The .p2align 2 before the constant pool absorbs the upstream
## growth, keeping the targets 4-byte aligned.

# CHECK-LABEL: <fn>:
# CHECK:       10: lrw16 r0,
# CHECK-NOT:       lrw32
# CHECK:       22: lrw16 r1,
# CHECK-NOT:       lrw32

    .text
    .globl fn
fn:
    br16 external
    .rept 6
    addu16 r0, r0, r1
    .endr
    lrw16 r0, [.LCPI0]
    .rept 6
    addu16 r0, r0, r1
    .endr
    br16 external
    lrw16 r1, [.LCPI1]
    .rept 6
    addu16 r0, r0, r1
    .endr
    .p2align 2
.LCPI0:
    .long 42
.LCPI1:
    .long 43
