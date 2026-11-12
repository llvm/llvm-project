# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump -d - | FileCheck %s
# Test that packets with extended immediates and new-value compare-jumps
# followed by alignment directives are encoded and decoded correctly.

# CHECK-LABEL: <test1>:
# CHECK: immext(#0x10000)
# CHECK-NEXT: r18 = ##0x10000
# CHECK-NEXT: if (!cmp.gtu(r1,r18.new)) jump:t
test1:
  .p2align 4
  {
    r18 = ##65536
    if (!cmp.gtu(r1,r18.new)) jump:t .L1
  }
  .p2align 4
.L1:
  nop

# CHECK-LABEL: <test2>:
# CHECK: immext(#0x20000)
# CHECK-NEXT: r19 = ##0x20000
# CHECK-NEXT: if (cmp.eq(r19.new,r2)) jump:nt
test2:
  .p2align 4
  {
    r19 = ##131072
    if (cmp.eq(r19.new,r2)) jump:nt .L2
  }
  .p2align 4
.L2:
  nop

# CHECK-LABEL: <test3>:
# CHECK: allocframe(#0x10)
# CHECK-NEXT: memd(r29+#0x0) = r19:18
# CHECK: immext(#0x10000)
# CHECK-NEXT: r18 = ##0x10000
# CHECK-NEXT: if (!cmp.gtu(r1,r18.new)) jump:t
test3:
  .p2align 4
  allocframe(#16)
  memd(r29+#0) = r19:18
  {
    r18 = ##65536
    if (!cmp.gtu(r1,r18.new)) jump:t .L3
  }
  .p2align 4
.L3:
  nop
