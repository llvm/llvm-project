# RUN: llvm-mc -arch=hexagon -filetype=asm %s 2>%t; FileCheck --implicit-check-not=error %s <%t

# Check that multiple changes to a predicate in a packet are caught.

	{ p0 = cmp.eq (r0, r0); p3:0 = r0 }
# CHECK: rror: register {{.+}} modified more than once

	{ p0 = cmp.eq (r0, r0); c4 = r0 }
# CHECK: rror: register {{.+}} modified more than once

	p3:0 = r9
# CHECK-NOT: rror: register {{.+}} modified more than once

# Multiple writes to the same predicate register are permitted:

	{ p0 = cmp.eq (r0, r0); p0 = and(p1, p2) }
# CHECK-NOT: rror: register {{.+}} modified more than once
