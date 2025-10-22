// RUN: llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

	.arch armv8-a+crypto

	aesd v0.16b, v2.16b
	eor v0.16b, v0.16b, v2.16b

# CHECK: 	aesd	v0.16b, v2.16b
# CHECK:        eor     v0.16b, v0.16b, v2.16b

	.arch armv8.1-a
        casa  w5, w7, [x20]
# CHECK:        casa    w5, w7, [x20]

	.arch armv8-a+lse
	casa  w5, w7, [x20]
# CHECK:        casa    w5, w7, [x20]

	.arch armv8.5-a+rng
	mrs   x0, rndr
	mrs   x0, rndrrs
# CHECK:        mrs     x0, RNDR
# CHECK:        mrs     x0, RNDRRS

	.arch armv9-a+cmpbr
	cbne x5, #31, lbl
# CHECK:        cbne x5, #31, lbl


	.arch armv9-a+fprcvt
	scvtf h1, s2
# CHECK:        scvtf h1, s2

	.arch armv9-a+f8f16mm
	fmmla v0.8h, v1.16b, v2.16b
# CHECK:        fmmla v0.8h, v1.16b, v2.16b

	.arch armv9-a+f8f32mm
	fmmla v0.4s, v1.16b, v2.16b
# CHECK:        fmmla v0.4s, v1.16b, v2.16b
