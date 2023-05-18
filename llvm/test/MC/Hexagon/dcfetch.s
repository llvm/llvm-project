# RUN: llvm-mc -triple=hexagon -filetype=obj -mno-pairing %s -o %t; llvm-objdump -d %t | FileCheck %s

# Check that DCFETCH is correctly shuffled.

	{ dcfetch(r2 + #0); r1 = memw(r2) }
# CHECK: 9402c000

# Bug 17424: This should be a legal packet
{
  P3 = SP1LOOP0(#8,R18)
  R7:6 = MEMUBH(R4++#4)
  R13:12 = VALIGNB(R11:10,R9:8,P2)
  DCFETCH(R5+#(8+0))
}
# CHECK-NOT: error:
