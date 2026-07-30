# RUN: llvm-mc -triple=hexagon -filetype=asm %s 2>%t; FileCheck %s <%t
#

# Check that proper packets are not wrongly flagged as invalid.

1-3-4-f:
	{
	       r3 = memub(r2++#1)
	       if (cmp.eq(r3.new,#0)) jump:nt .
	       jumpr lr
	       r4 = #4
	}
# CHECK-NOT: rror: invalid instruction packet

1-3-f-f:
        {
                r3 = memub(r2++#1)
                if (cmp.eq(r3.new,#0)) jump:nt .
                r5 = #5
                r4 = #4
        }
# CHECK-NOT: rror: invalid instruction packet

# Special case of a fat packet that will slim when a compound is formed.
3-3-8-c:
   { LOOP0(3-3-8-c, R7)
     P0 = CMP.GT(R7, #0)
     IF (!P0.NEW) JUMP:NT .
     R21:20 = MEMD(R0+#16)
     R23:22 = MEMD(R0+#24)
   }
# CHECK-NOT: rror: invalid instruction packet

1-f-f-f:
        {
                r3 = #3
                if (cmp.eq(r3.new,#0)) jump:nt .
                r5 = #5
                r4 = #4
        }
# CHECK-NOT: rror: invalid instruction packet

4:
        jumpr lr
# CHECK-NOT: rror: invalid instruction packet

f-f-f-f:
        {
                r3 = #3
                r2 = #2
                r5 = #5
                r4 = #4
        }
# CHECK-NOT: rror: invalid instruction packet

