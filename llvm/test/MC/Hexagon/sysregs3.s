# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump -d - | FileCheck %s
#

# Verify exceptions to the grouping rules for some registers.

	{ r6=ssr; r0=memw(r0) }
# CHECK: { r6 = ssr
	{ r7:6=ccr:ssr; r1:0=memd(r0) }
# CHECK: { r7:6 = s7:6
	{ ssr=r6; r0=memw(r0) }
# CHECK: { ssr = r6
	{ s7:6=r7:6; r1:0=memd(r0) }
# CHECK: { s7:6 = r7:6
