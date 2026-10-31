# RUN: llvm-mc -triple=hexagon-unknown-linux-musl -filetype=obj %s \
# RUN:   | llvm-objdump -d - | FileCheck %s

# When a .p2align directive following a packet with a conditional
# compare-and-jump forces MC relaxation of the packet (to add pad nops
# so that the following code is aligned), the branch fixup must be
# updated after re-encoding. Regression test for issue #163851: the
# recomputed fixup was silently dropped, leaving an invalid packet
# encoding (missing packet-end bit) and an incorrect branch target.

    .text
    .globl foo
    .p2align 4
    .type foo,@function
foo:
    { r0 = r1
      if (cmp.eq(r0.new,#0)) jump:nt .Lend }
    .p2align 5
.Lloop:
    { r0 = add(r0,#1) }
    { jump .Lloop }
.Lend:
    { jumpr r31 }
    .size foo, .-foo

# CHECK-LABEL: <foo>:
# CHECK:      r0 = r1
# CHECK-NEXT: if (cmp.eq(r0.new,#0x0)) jump:nt {{0x[0-9a-f]+}} <foo+0x28>
# CHECK-NOT:  <unknown>
