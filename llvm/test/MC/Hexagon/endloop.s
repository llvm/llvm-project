# RUN: not llvm-mc -triple=hexagon -filetype=asm %s 2>&1 | FileCheck %s

# Check that a branch in an end-loop packet is caught.

1:
{
	r0 = #1
	p0 = cmp.eq (r1, r2)
	if (p0) jump 1b
}:endloop0

2:
{
        r0 = #1
        p0 = cmp.eq (r1, r2)
        if (p0) jump 2b
}:endloop1

# CHECK: rror: packet marked with `:endloop{{.}}' cannot contain instructions that modify register
