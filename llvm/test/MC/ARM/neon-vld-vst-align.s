@ RUN: not llvm-mc -triple=thumbv7-apple-darwin -show-encoding < %s > %t 2> %t.err
@ RUN: FileCheck < %t %s
@ RUN: FileCheck --check-prefix=CHECK-ERRORS < %t.err %s

	vld1.8	{d0}, [r4]
	vld1.8	{d0}, [r4:16]
	vld1.8	{d0}, [r4:32]
	vld1.8	{d0}, [r4:64]
	vld1.8	{d0}, [r4:128]
	vld1.8	{d0}, [r4:256]

@ CHECK: vld1.8	{d0}, [r4]              @ encoding: [0x24,0xf9,0x0f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0}, [r4:16]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0}, [r4:32]
@ CHECK-ERRORS:                           ^
@ CHECK: vld1.8	{d0}, [r4:64]           @ encoding: [0x24,0xf9,0x1f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0}, [r4:128]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0}, [r4:256]
@ CHECK-ERRORS:                           ^

	vld1.8	{d0}, [r4]!
	vld1.8	{d0}, [r4:16]!
	vld1.8	{d0}, [r4:32]!
	vld1.8	{d0}, [r4:64]!
	vld1.8	{d0}, [r4:128]!
	vld1.8	{d0}, [r4:256]!

@ CHECK: vld1.8	{d0}, [r4]!             @ encoding: [0x24,0xf9,0x0d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0}, [r4:16]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0}, [r4:32]!
@ CHECK-ERRORS:                           ^
@ CHECK: vld1.8	{d0}, [r4:64]!          @ encoding: [0x24,0xf9,0x1d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0}, [r4:128]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0}, [r4:256]!
@ CHECK-ERRORS:                           ^

	vld1.8	{d0}, [r4], r6
	vld1.8	{d0}, [r4:16], r6
	vld1.8	{d0}, [r4:32], r6
	vld1.8	{d0}, [r4:64], r6
	vld1.8	{d0}, [r4:128], r6
	vld1.8	{d0}, [r4:256], r6

@ CHECK: vld1.8	{d0}, [r4], r6          @ encoding: [0x24,0xf9,0x06,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0}, [r4:16], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0}, [r4:32], r6
@ CHECK-ERRORS:                           ^
@ CHECK: vld1.8	{d0}, [r4:64], r6       @ encoding: [0x24,0xf9,0x16,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0}, [r4:128], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0}, [r4:256], r6
@ CHECK-ERRORS:                           ^

	vld1.8	{d0, d1}, [r4]
	vld1.8	{d0, d1}, [r4:16]
	vld1.8	{d0, d1}, [r4:32]
	vld1.8	{d0, d1}, [r4:64]
	vld1.8	{d0, d1}, [r4:128]
	vld1.8	{d0, d1}, [r4:256]

@ CHECK: vld1.8	{d0, d1}, [r4]          @ encoding: [0x24,0xf9,0x0f,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vld1.8	{d0, d1}, [r4:64]       @ encoding: [0x24,0xf9,0x1f,0x0a]
@ CHECK: vld1.8	{d0, d1}, [r4:128]      @ encoding: [0x24,0xf9,0x2f,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1}, [r4:256]
@ CHECK-ERRORS:                               ^

	vld1.8	{d0, d1}, [r4]!
	vld1.8	{d0, d1}, [r4:16]!
	vld1.8	{d0, d1}, [r4:32]!
	vld1.8	{d0, d1}, [r4:64]!
	vld1.8	{d0, d1}, [r4:128]!
	vld1.8	{d0, d1}, [r4:256]!

@ CHECK: vld1.8	{d0, d1}, [r4]!         @ encoding: [0x24,0xf9,0x0d,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vld1.8	{d0, d1}, [r4:64]!      @ encoding: [0x24,0xf9,0x1d,0x0a]
@ CHECK: vld1.8	{d0, d1}, [r4:128]!     @ encoding: [0x24,0xf9,0x2d,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vld1.8	{d0, d1}, [r4], r6
	vld1.8	{d0, d1}, [r4:16], r6
	vld1.8	{d0, d1}, [r4:32], r6
	vld1.8	{d0, d1}, [r4:64], r6
	vld1.8	{d0, d1}, [r4:128], r6
	vld1.8	{d0, d1}, [r4:256], r6

@ CHECK: vld1.8	{d0, d1}, [r4], r6      @ encoding: [0x24,0xf9,0x06,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vld1.8	{d0, d1}, [r4:64], r6   @ encoding: [0x24,0xf9,0x16,0x0a]
@ CHECK: vld1.8	{d0, d1}, [r4:128], r6  @ encoding: [0x24,0xf9,0x26,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vld1.8	{d0, d1, d2}, [r4]
	vld1.8	{d0, d1, d2}, [r4:16]
	vld1.8	{d0, d1, d2}, [r4:32]
	vld1.8	{d0, d1, d2}, [r4:64]
	vld1.8	{d0, d1, d2}, [r4:128]
	vld1.8	{d0, d1, d2}, [r4:256]

@ CHECK: vld1.8	{d0, d1, d2}, [r4]      @ encoding: [0x24,0xf9,0x0f,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.8	{d0, d1, d2}, [r4:64]   @ encoding: [0x24,0xf9,0x1f,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld1.8	{d0, d1, d2}, [r4]!
	vld1.8	{d0, d1, d2}, [r4:16]!
	vld1.8	{d0, d1, d2}, [r4:32]!
	vld1.8	{d0, d1, d2}, [r4:64]!
	vld1.8	{d0, d1, d2}, [r4:128]!
	vld1.8	{d0, d1, d2}, [r4:256]!

@ CHECK: vld1.8	{d0, d1, d2}, [r4]!     @ encoding: [0x24,0xf9,0x0d,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.8	{d0, d1, d2}, [r4:64]!  @ encoding: [0x24,0xf9,0x1d,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld1.8	{d0, d1, d2}, [r4], r6
	vld1.8	{d0, d1, d2}, [r4:16], r6
	vld1.8	{d0, d1, d2}, [r4:32], r6
	vld1.8	{d0, d1, d2}, [r4:64], r6
	vld1.8	{d0, d1, d2}, [r4:128], r6
	vld1.8	{d0, d1, d2}, [r4:256], r6

@ CHECK: vld1.8	{d0, d1, d2}, [r4], r6  @ encoding: [0x24,0xf9,0x06,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.8	{d0, d1, d2}, [r4:64], r6 @ encoding: [0x24,0xf9,0x16,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld1.8	{d0, d1, d2, d3}, [r4]
	vld1.8	{d0, d1, d2, d3}, [r4:16]
	vld1.8	{d0, d1, d2, d3}, [r4:32]
	vld1.8	{d0, d1, d2, d3}, [r4:64]
	vld1.8	{d0, d1, d2, d3}, [r4:128]
	vld1.8	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vld1.8	{d0, d1, d2, d3}, [r4]  @ encoding: [0x24,0xf9,0x0f,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vld1.8	{d0, d1, d2, d3}, [r4:64] @ encoding: [0x24,0xf9,0x1f,0x02]
@ CHECK: vld1.8	{d0, d1, d2, d3}, [r4:128] @ encoding: [0x24,0xf9,0x2f,0x02]
@ CHECK: vld1.8	{d0, d1, d2, d3}, [r4:256] @ encoding: [0x24,0xf9,0x3f,0x02]

	vld1.8	{d0, d1, d2, d3}, [r4]!
	vld1.8	{d0, d1, d2, d3}, [r4:16]!
	vld1.8	{d0, d1, d2, d3}, [r4:32]!
	vld1.8	{d0, d1, d2, d3}, [r4:64]!
	vld1.8	{d0, d1, d2, d3}, [r4:128]!
	vld1.8	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vld1.8	{d0, d1, d2, d3}, [r4]! @ encoding: [0x24,0xf9,0x0d,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vld1.8	{d0, d1, d2, d3}, [r4:64]! @ encoding: [0x24,0xf9,0x1d,0x02]
@ CHECK: vld1.8	{d0, d1, d2, d3}, [r4:128]! @ encoding: [0x24,0xf9,0x2d,0x02]
@ CHECK: vld1.8	{d0, d1, d2, d3}, [r4:256]! @ encoding: [0x24,0xf9,0x3d,0x02]

	vld1.8	{d0, d1, d2, d3}, [r4], r6
	vld1.8	{d0, d1, d2, d3}, [r4:16], r6
	vld1.8	{d0, d1, d2, d3}, [r4:32], r6
	vld1.8	{d0, d1, d2, d3}, [r4:64], r6
	vld1.8	{d0, d1, d2, d3}, [r4:128], r6
	vld1.8	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vld1.8	{d0, d1, d2, d3}, [r4], r6 @ encoding: [0x24,0xf9,0x06,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.8  {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vld1.8	{d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x24,0xf9,0x16,0x02]
@ CHECK: vld1.8	{d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x24,0xf9,0x26,0x02]
@ CHECK: vld1.8	{d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x24,0xf9,0x36,0x02]

	vld1.8	{d0[2]}, [r4]
	vld1.8	{d0[2]}, [r4:16]
	vld1.8	{d0[2]}, [r4:32]
	vld1.8	{d0[2]}, [r4:64]
	vld1.8	{d0[2]}, [r4:128]
	vld1.8	{d0[2]}, [r4:256]

@ CHECK: vld1.8	{d0[2]}, [r4]           @ encoding: [0xa4,0xf9,0x4f,0x00]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:16]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:32]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:64]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:128]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:256]
@ CHECK-ERRORS:                              ^

	vld1.8	{d0[2]}, [r4]!
	vld1.8	{d0[2]}, [r4:16]!
	vld1.8	{d0[2]}, [r4:32]!
	vld1.8	{d0[2]}, [r4:64]!
	vld1.8	{d0[2]}, [r4:128]!
	vld1.8	{d0[2]}, [r4:256]!

@ CHECK: vld1.8	{d0[2]}, [r4]!          @ encoding: [0xa4,0xf9,0x4d,0x00]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:16]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:32]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:64]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:128]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:256]!
@ CHECK-ERRORS:                              ^

	vld1.8	{d0[2]}, [r4], r6
	vld1.8	{d0[2]}, [r4:16], r6
	vld1.8	{d0[2]}, [r4:32], r6
	vld1.8	{d0[2]}, [r4:64], r6
	vld1.8	{d0[2]}, [r4:128], r6
	vld1.8	{d0[2]}, [r4:256], r6

@ CHECK: vld1.8	{d0[2]}, [r4], r6       @ encoding: [0xa4,0xf9,0x46,0x00]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:16], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:32], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:64], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:128], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[2]}, [r4:256], r6
@ CHECK-ERRORS:                              ^

	vld1.8	{d0[]}, [r4]
	vld1.8	{d0[]}, [r4:16]
	vld1.8	{d0[]}, [r4:32]
	vld1.8	{d0[]}, [r4:64]
	vld1.8	{d0[]}, [r4:128]
	vld1.8	{d0[]}, [r4:256]

@ CHECK: vld1.8	{d0[]}, [r4]            @ encoding: [0xa4,0xf9,0x0f,0x0c]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:16]
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:32]
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:64]
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:128]
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:256]
@ CHECK-ERRORS:                             ^

	vld1.8	{d0[]}, [r4]!
	vld1.8	{d0[]}, [r4:16]!
	vld1.8	{d0[]}, [r4:32]!
	vld1.8	{d0[]}, [r4:64]!
	vld1.8	{d0[]}, [r4:128]!
	vld1.8	{d0[]}, [r4:256]!

@ CHECK: vld1.8	{d0[]}, [r4]!           @ encoding: [0xa4,0xf9,0x0d,0x0c]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:16]!
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:32]!
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:64]!
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:128]!
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:256]!
@ CHECK-ERRORS:                             ^

	vld1.8	{d0[]}, [r4], r6
	vld1.8	{d0[]}, [r4:16], r6
	vld1.8	{d0[]}, [r4:32], r6
	vld1.8	{d0[]}, [r4:64], r6
	vld1.8	{d0[]}, [r4:128], r6
	vld1.8	{d0[]}, [r4:256], r6

@ CHECK: vld1.8	{d0[]}, [r4], r6        @ encoding: [0xa4,0xf9,0x06,0x0c]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:16], r6
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:32], r6
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:64], r6
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:128], r6
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[]}, [r4:256], r6
@ CHECK-ERRORS:                             ^

	vld1.8	{d0[], d1[]}, [r4]
	vld1.8	{d0[], d1[]}, [r4:16]
	vld1.8	{d0[], d1[]}, [r4:32]
	vld1.8	{d0[], d1[]}, [r4:64]
	vld1.8	{d0[], d1[]}, [r4:128]
	vld1.8	{d0[], d1[]}, [r4:256]

@ CHECK: vld1.8	{d0[], d1[]}, [r4]      @ encoding: [0xa4,0xf9,0x2f,0x0c]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:64]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld1.8	{d0[], d1[]}, [r4]!
	vld1.8	{d0[], d1[]}, [r4:16]!
	vld1.8	{d0[], d1[]}, [r4:32]!
	vld1.8	{d0[], d1[]}, [r4:64]!
	vld1.8	{d0[], d1[]}, [r4:128]!
	vld1.8	{d0[], d1[]}, [r4:256]!

@ CHECK: vld1.8	{d0[], d1[]}, [r4]!     @ encoding: [0xa4,0xf9,0x2d,0x0c]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:64]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld1.8	{d0[], d1[]}, [r4], r6
	vld1.8	{d0[], d1[]}, [r4:16], r6
	vld1.8	{d0[], d1[]}, [r4:32], r6
	vld1.8	{d0[], d1[]}, [r4:64], r6
	vld1.8	{d0[], d1[]}, [r4:128], r6
	vld1.8	{d0[], d1[]}, [r4:256], r6

@ CHECK: vld1.8	{d0[], d1[]}, [r4], r6  @ encoding: [0xa4,0xf9,0x26,0x0c]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:64], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld1.8  {d0[], d1[]}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld1.16	{d0}, [r4]
	vld1.16	{d0}, [r4:16]
	vld1.16	{d0}, [r4:32]
	vld1.16	{d0}, [r4:64]
	vld1.16	{d0}, [r4:128]
	vld1.16	{d0}, [r4:256]

@ CHECK: vld1.16 {d0}, [r4]              @ encoding: [0x24,0xf9,0x4f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0}, [r4:16]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0}, [r4:32]
@ CHECK-ERRORS:                           ^
@ CHECK: vld1.16 {d0}, [r4:64]           @ encoding: [0x24,0xf9,0x5f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0}, [r4:128]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0}, [r4:256]
@ CHECK-ERRORS:                           ^

	vld1.16	{d0}, [r4]!
	vld1.16	{d0}, [r4:16]!
	vld1.16	{d0}, [r4:32]!
	vld1.16	{d0}, [r4:64]!
	vld1.16	{d0}, [r4:128]!
	vld1.16	{d0}, [r4:256]!

@ CHECK: vld1.16 {d0}, [r4]!             @ encoding: [0x24,0xf9,0x4d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0}, [r4:16]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0}, [r4:32]!
@ CHECK-ERRORS:                           ^
@ CHECK: vld1.16 {d0}, [r4:64]!          @ encoding: [0x24,0xf9,0x5d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0}, [r4:128]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0}, [r4:256]!
@ CHECK-ERRORS:                           ^

	vld1.16	{d0}, [r4], r6
	vld1.16	{d0}, [r4:16], r6
	vld1.16	{d0}, [r4:32], r6
	vld1.16	{d0}, [r4:64], r6
	vld1.16	{d0}, [r4:128], r6
	vld1.16	{d0}, [r4:256], r6

@ CHECK: vld1.16 {d0}, [r4], r6          @ encoding: [0x24,0xf9,0x46,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0}, [r4:16], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0}, [r4:32], r6
@ CHECK-ERRORS:                           ^
@ CHECK: vld1.16 {d0}, [r4:64], r6       @ encoding: [0x24,0xf9,0x56,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0}, [r4:128], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0}, [r4:256], r6
@ CHECK-ERRORS:                           ^

	vld1.16	{d0, d1}, [r4]
	vld1.16	{d0, d1}, [r4:16]
	vld1.16	{d0, d1}, [r4:32]
	vld1.16	{d0, d1}, [r4:64]
	vld1.16	{d0, d1}, [r4:128]
	vld1.16	{d0, d1}, [r4:256]

@ CHECK: vld1.16 {d0, d1}, [r4]          @ encoding: [0x24,0xf9,0x4f,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vld1.16 {d0, d1}, [r4:64]       @ encoding: [0x24,0xf9,0x5f,0x0a]
@ CHECK: vld1.16 {d0, d1}, [r4:128]      @ encoding: [0x24,0xf9,0x6f,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1}, [r4:256]
@ CHECK-ERRORS:                               ^

	vld1.16	{d0, d1}, [r4]!
	vld1.16	{d0, d1}, [r4:16]!
	vld1.16	{d0, d1}, [r4:32]!
	vld1.16	{d0, d1}, [r4:64]!
	vld1.16	{d0, d1}, [r4:128]!
	vld1.16	{d0, d1}, [r4:256]!

@ CHECK: vld1.16 {d0, d1}, [r4]!         @ encoding: [0x24,0xf9,0x4d,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vld1.16 {d0, d1}, [r4:64]!      @ encoding: [0x24,0xf9,0x5d,0x0a]
@ CHECK: vld1.16 {d0, d1}, [r4:128]!     @ encoding: [0x24,0xf9,0x6d,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vld1.16	{d0, d1}, [r4], r6
	vld1.16	{d0, d1}, [r4:16], r6
	vld1.16	{d0, d1}, [r4:32], r6
	vld1.16	{d0, d1}, [r4:64], r6
	vld1.16	{d0, d1}, [r4:128], r6
	vld1.16	{d0, d1}, [r4:256], r6

@ CHECK: vld1.16 {d0, d1}, [r4], r6      @ encoding: [0x24,0xf9,0x46,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vld1.16 {d0, d1}, [r4:64], r6   @ encoding: [0x24,0xf9,0x56,0x0a]
@ CHECK: vld1.16 {d0, d1}, [r4:128], r6  @ encoding: [0x24,0xf9,0x66,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vld1.16	{d0, d1, d2}, [r4]
	vld1.16	{d0, d1, d2}, [r4:16]
	vld1.16	{d0, d1, d2}, [r4:32]
	vld1.16	{d0, d1, d2}, [r4:64]
	vld1.16	{d0, d1, d2}, [r4:128]
	vld1.16	{d0, d1, d2}, [r4:256]

@ CHECK: vld1.16 {d0, d1, d2}, [r4]      @ encoding: [0x24,0xf9,0x4f,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.16 {d0, d1, d2}, [r4:64]   @ encoding: [0x24,0xf9,0x5f,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld1.16	{d0, d1, d2}, [r4]!
	vld1.16	{d0, d1, d2}, [r4:16]!
	vld1.16	{d0, d1, d2}, [r4:32]!
	vld1.16	{d0, d1, d2}, [r4:64]!
	vld1.16	{d0, d1, d2}, [r4:128]!
	vld1.16	{d0, d1, d2}, [r4:256]!

@ CHECK: vld1.16 {d0, d1, d2}, [r4]!     @ encoding: [0x24,0xf9,0x4d,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.16 {d0, d1, d2}, [r4:64]!  @ encoding: [0x24,0xf9,0x5d,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld1.16	{d0, d1, d2}, [r4], r6
	vld1.16	{d0, d1, d2}, [r4:16], r6
	vld1.16	{d0, d1, d2}, [r4:32], r6
	vld1.16	{d0, d1, d2}, [r4:64], r6
	vld1.16	{d0, d1, d2}, [r4:128], r6
	vld1.16	{d0, d1, d2}, [r4:256], r6

@ CHECK: vld1.16 {d0, d1, d2}, [r4], r6  @ encoding: [0x24,0xf9,0x46,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.16 {d0, d1, d2}, [r4:64], r6 @ encoding: [0x24,0xf9,0x56,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld1.16	{d0, d1, d2, d3}, [r4]
	vld1.16	{d0, d1, d2, d3}, [r4:16]
	vld1.16	{d0, d1, d2, d3}, [r4:32]
	vld1.16	{d0, d1, d2, d3}, [r4:64]
	vld1.16	{d0, d1, d2, d3}, [r4:128]
	vld1.16	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vld1.16 {d0, d1, d2, d3}, [r4]  @ encoding: [0x24,0xf9,0x4f,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vld1.16 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x24,0xf9,0x5f,0x02]
@ CHECK: vld1.16 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x24,0xf9,0x6f,0x02]
@ CHECK: vld1.16 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x24,0xf9,0x7f,0x02]

	vld1.16	{d0, d1, d2, d3}, [r4]!
	vld1.16	{d0, d1, d2, d3}, [r4:16]!
	vld1.16	{d0, d1, d2, d3}, [r4:32]!
	vld1.16	{d0, d1, d2, d3}, [r4:64]!
	vld1.16	{d0, d1, d2, d3}, [r4:128]!
	vld1.16	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vld1.16 {d0, d1, d2, d3}, [r4]! @ encoding: [0x24,0xf9,0x4d,0x02]
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vld1.16 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x24,0xf9,0x5d,0x02]
@ CHECK: vld1.16 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x24,0xf9,0x6d,0x02]
@ CHECK: vld1.16 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x24,0xf9,0x7d,0x02]

	vld1.16	{d0, d1, d2, d3}, [r4], r6
	vld1.16	{d0, d1, d2, d3}, [r4:16], r6
	vld1.16	{d0, d1, d2, d3}, [r4:32], r6
	vld1.16	{d0, d1, d2, d3}, [r4:64], r6
	vld1.16	{d0, d1, d2, d3}, [r4:128], r6
	vld1.16	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vld1.16 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x24,0xf9,0x46,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.16 {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vld1.16 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x24,0xf9,0x56,0x02]
@ CHECK: vld1.16 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x24,0xf9,0x66,0x02]
@ CHECK: vld1.16 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x24,0xf9,0x76,0x02]

	vld1.16	{d0[2]}, [r4]
	vld1.16	{d0[2]}, [r4:16]
	vld1.16	{d0[2]}, [r4:32]
	vld1.16	{d0[2]}, [r4:64]
	vld1.16	{d0[2]}, [r4:128]
	vld1.16	{d0[2]}, [r4:256]

@ CHECK: vld1.16 {d0[2]}, [r4]           @ encoding: [0xa4,0xf9,0x8f,0x04]
@ CHECK: vld1.16 {d0[2]}, [r4:16]        @ encoding: [0xa4,0xf9,0x9f,0x04]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[2]}, [r4:32]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[2]}, [r4:64]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[2]}, [r4:128]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[2]}, [r4:256]
@ CHECK-ERRORS:                              ^

	vld1.16	{d0[2]}, [r4]!
	vld1.16	{d0[2]}, [r4:16]!
	vld1.16	{d0[2]}, [r4:32]!
	vld1.16	{d0[2]}, [r4:64]!
	vld1.16	{d0[2]}, [r4:128]!
	vld1.16	{d0[2]}, [r4:256]!

@ CHECK: vld1.16 {d0[2]}, [r4]!          @ encoding: [0xa4,0xf9,0x8d,0x04]
@ CHECK: vld1.16 {d0[2]}, [r4:16]!       @ encoding: [0xa4,0xf9,0x9d,0x04]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[2]}, [r4:32]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[2]}, [r4:64]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[2]}, [r4:128]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[2]}, [r4:256]!
@ CHECK-ERRORS:                              ^

	vld1.16	{d0[2]}, [r4], r6
	vld1.16	{d0[2]}, [r4:16], r6
	vld1.16	{d0[2]}, [r4:32], r6
	vld1.16	{d0[2]}, [r4:64], r6
	vld1.16	{d0[2]}, [r4:128], r6
	vld1.16	{d0[2]}, [r4:256], r6

@ CHECK: vld1.16 {d0[2]}, [r4], r6       @ encoding: [0xa4,0xf9,0x86,0x04]
@ CHECK: vld1.16 {d0[2]}, [r4:16], r6    @ encoding: [0xa4,0xf9,0x96,0x04]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[2]}, [r4:32], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[2]}, [r4:64], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[2]}, [r4:128], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[2]}, [r4:256], r6
@ CHECK-ERRORS:                              ^

	vld1.16	{d0[]}, [r4]
	vld1.16	{d0[]}, [r4:16]
	vld1.16	{d0[]}, [r4:32]
	vld1.16	{d0[]}, [r4:64]
	vld1.16	{d0[]}, [r4:128]
	vld1.16	{d0[]}, [r4:256]

@ CHECK: vld1.16 {d0[]}, [r4]            @ encoding: [0xa4,0xf9,0x4f,0x0c]
@ CHECK: vld1.16 {d0[]}, [r4:16]         @ encoding: [0xa4,0xf9,0x5f,0x0c]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[]}, [r4:32]
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[]}, [r4:64]
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[]}, [r4:128]
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[]}, [r4:256]
@ CHECK-ERRORS:                             ^

	vld1.16	{d0[]}, [r4]!
	vld1.16	{d0[]}, [r4:16]!
	vld1.16	{d0[]}, [r4:32]!
	vld1.16	{d0[]}, [r4:64]!
	vld1.16	{d0[]}, [r4:128]!
	vld1.16	{d0[]}, [r4:256]!

@ CHECK: vld1.16 {d0[]}, [r4]!           @ encoding: [0xa4,0xf9,0x4d,0x0c]
@ CHECK: vld1.16 {d0[]}, [r4:16]!        @ encoding: [0xa4,0xf9,0x5d,0x0c]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[]}, [r4:32]!
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[]}, [r4:64]!
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[]}, [r4:128]!
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[]}, [r4:256]!
@ CHECK-ERRORS:                             ^

	vld1.16	{d0[]}, [r4], r6
	vld1.16	{d0[]}, [r4:16], r6
	vld1.16	{d0[]}, [r4:32], r6
	vld1.16	{d0[]}, [r4:64], r6
	vld1.16	{d0[]}, [r4:128], r6
	vld1.16	{d0[]}, [r4:256], r6

@ CHECK: vld1.16 {d0[]}, [r4], r6        @ encoding: [0xa4,0xf9,0x46,0x0c]
@ CHECK: vld1.16 {d0[]}, [r4:16], r6     @ encoding: [0xa4,0xf9,0x56,0x0c]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[]}, [r4:32], r6
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[]}, [r4:64], r6
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[]}, [r4:128], r6
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[]}, [r4:256], r6
@ CHECK-ERRORS:                             ^

	vld1.16	{d0[], d1[]}, [r4]
	vld1.16	{d0[], d1[]}, [r4:16]
	vld1.16	{d0[], d1[]}, [r4:32]
	vld1.16	{d0[], d1[]}, [r4:64]
	vld1.16	{d0[], d1[]}, [r4:128]
	vld1.16	{d0[], d1[]}, [r4:256]

@ CHECK: vld1.16 {d0[], d1[]}, [r4]      @ encoding: [0xa4,0xf9,0x6f,0x0c]
@ CHECK: vld1.16 {d0[], d1[]}, [r4:16]   @ encoding: [0xa4,0xf9,0x7f,0x0c]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[], d1[]}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[], d1[]}, [r4:64]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[], d1[]}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[], d1[]}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld1.16	{d0[], d1[]}, [r4]!
	vld1.16	{d0[], d1[]}, [r4:16]!
	vld1.16	{d0[], d1[]}, [r4:32]!
	vld1.16	{d0[], d1[]}, [r4:64]!
	vld1.16	{d0[], d1[]}, [r4:128]!
	vld1.16	{d0[], d1[]}, [r4:256]!

@ CHECK: vld1.16 {d0[], d1[]}, [r4]!     @ encoding: [0xa4,0xf9,0x6d,0x0c]
@ CHECK: vld1.16 {d0[], d1[]}, [r4:16]!  @ encoding: [0xa4,0xf9,0x7d,0x0c]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[], d1[]}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[], d1[]}, [r4:64]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[], d1[]}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[], d1[]}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld1.16	{d0[], d1[]}, [r4], r6
	vld1.16	{d0[], d1[]}, [r4:16], r6
	vld1.16	{d0[], d1[]}, [r4:32], r6
	vld1.16	{d0[], d1[]}, [r4:64], r6
	vld1.16	{d0[], d1[]}, [r4:128], r6
	vld1.16	{d0[], d1[]}, [r4:256], r6

@ CHECK: vld1.16 {d0[], d1[]}, [r4], r6  @ encoding: [0xa4,0xf9,0x66,0x0c]
@ CHECK: vld1.16 {d0[], d1[]}, [r4:16], r6 @ encoding: [0xa4,0xf9,0x76,0x0c]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[], d1[]}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[], d1[]}, [r4:64], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[], d1[]}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld1.16 {d0[], d1[]}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld1.32	{d0}, [r4]
	vld1.32	{d0}, [r4:16]
	vld1.32	{d0}, [r4:32]
	vld1.32	{d0}, [r4:64]
	vld1.32	{d0}, [r4:128]
	vld1.32	{d0}, [r4:256]

@ CHECK: vld1.32 {d0}, [r4]              @ encoding: [0x24,0xf9,0x8f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0}, [r4:16]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0}, [r4:32]
@ CHECK-ERRORS:                           ^
@ CHECK: vld1.32 {d0}, [r4:64]           @ encoding: [0x24,0xf9,0x9f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0}, [r4:128]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0}, [r4:256]
@ CHECK-ERRORS:                           ^

	vld1.32	{d0}, [r4]!
	vld1.32	{d0}, [r4:16]!
	vld1.32	{d0}, [r4:32]!
	vld1.32	{d0}, [r4:64]!
	vld1.32	{d0}, [r4:128]!
	vld1.32	{d0}, [r4:256]!

@ CHECK: vld1.32 {d0}, [r4]!             @ encoding: [0x24,0xf9,0x8d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0}, [r4:16]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0}, [r4:32]!
@ CHECK-ERRORS:                           ^
@ CHECK: vld1.32 {d0}, [r4:64]!          @ encoding: [0x24,0xf9,0x9d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0}, [r4:128]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0}, [r4:256]!
@ CHECK-ERRORS:                           ^

	vld1.32	{d0}, [r4], r6
	vld1.32	{d0}, [r4:16], r6
	vld1.32	{d0}, [r4:32], r6
	vld1.32	{d0}, [r4:64], r6
	vld1.32	{d0}, [r4:128], r6
	vld1.32	{d0}, [r4:256], r6

@ CHECK: vld1.32 {d0}, [r4], r6          @ encoding: [0x24,0xf9,0x86,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0}, [r4:16], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0}, [r4:32], r6
@ CHECK-ERRORS:                           ^
@ CHECK: vld1.32 {d0}, [r4:64], r6       @ encoding: [0x24,0xf9,0x96,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0}, [r4:128], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0}, [r4:256], r6
@ CHECK-ERRORS:                           ^

	vld1.32	{d0, d1}, [r4]
	vld1.32	{d0, d1}, [r4:16]
	vld1.32	{d0, d1}, [r4:32]
	vld1.32	{d0, d1}, [r4:64]
	vld1.32	{d0, d1}, [r4:128]
	vld1.32	{d0, d1}, [r4:256]

@ CHECK: vld1.32 {d0, d1}, [r4]          @ encoding: [0x24,0xf9,0x8f,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vld1.32 {d0, d1}, [r4:64]       @ encoding: [0x24,0xf9,0x9f,0x0a]
@ CHECK: vld1.32 {d0, d1}, [r4:128]      @ encoding: [0x24,0xf9,0xaf,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1}, [r4:256]
@ CHECK-ERRORS:                               ^

	vld1.32	{d0, d1}, [r4]!
	vld1.32	{d0, d1}, [r4:16]!
	vld1.32	{d0, d1}, [r4:32]!
	vld1.32	{d0, d1}, [r4:64]!
	vld1.32	{d0, d1}, [r4:128]!
	vld1.32	{d0, d1}, [r4:256]!

@ CHECK: vld1.32 {d0, d1}, [r4]!         @ encoding: [0x24,0xf9,0x8d,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vld1.32 {d0, d1}, [r4:64]!      @ encoding: [0x24,0xf9,0x9d,0x0a]
@ CHECK: vld1.32 {d0, d1}, [r4:128]!     @ encoding: [0x24,0xf9,0xad,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vld1.32	{d0, d1}, [r4], r6
	vld1.32	{d0, d1}, [r4:16], r6
	vld1.32	{d0, d1}, [r4:32], r6
	vld1.32	{d0, d1}, [r4:64], r6
	vld1.32	{d0, d1}, [r4:128], r6
	vld1.32	{d0, d1}, [r4:256], r6

@ CHECK: vld1.32 {d0, d1}, [r4], r6      @ encoding: [0x24,0xf9,0x86,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vld1.32 {d0, d1}, [r4:64], r6   @ encoding: [0x24,0xf9,0x96,0x0a]
@ CHECK: vld1.32 {d0, d1}, [r4:128], r6  @ encoding: [0x24,0xf9,0xa6,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vld1.32	{d0, d1, d2}, [r4]
	vld1.32	{d0, d1, d2}, [r4:16]
	vld1.32	{d0, d1, d2}, [r4:32]
	vld1.32	{d0, d1, d2}, [r4:64]
	vld1.32	{d0, d1, d2}, [r4:128]
	vld1.32	{d0, d1, d2}, [r4:256]

@ CHECK: vld1.32 {d0, d1, d2}, [r4]      @ encoding: [0x24,0xf9,0x8f,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.32 {d0, d1, d2}, [r4:64]   @ encoding: [0x24,0xf9,0x9f,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld1.32	{d0, d1, d2}, [r4]!
	vld1.32	{d0, d1, d2}, [r4:16]!
	vld1.32	{d0, d1, d2}, [r4:32]!
	vld1.32	{d0, d1, d2}, [r4:64]!
	vld1.32	{d0, d1, d2}, [r4:128]!
	vld1.32	{d0, d1, d2}, [r4:256]!

@ CHECK: vld1.32 {d0, d1, d2}, [r4]!     @ encoding: [0x24,0xf9,0x8d,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.32 {d0, d1, d2}, [r4:64]!  @ encoding: [0x24,0xf9,0x9d,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld1.32	{d0, d1, d2}, [r4], r6
	vld1.32	{d0, d1, d2}, [r4:16], r6
	vld1.32	{d0, d1, d2}, [r4:32], r6
	vld1.32	{d0, d1, d2}, [r4:64], r6
	vld1.32	{d0, d1, d2}, [r4:128], r6
	vld1.32	{d0, d1, d2}, [r4:256], r6

@ CHECK: vld1.32 {d0, d1, d2}, [r4], r6  @ encoding: [0x24,0xf9,0x86,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.32 {d0, d1, d2}, [r4:64], r6 @ encoding: [0x24,0xf9,0x96,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld1.32	{d0, d1, d2, d3}, [r4]
	vld1.32	{d0, d1, d2, d3}, [r4:16]
	vld1.32	{d0, d1, d2, d3}, [r4:32]
	vld1.32	{d0, d1, d2, d3}, [r4:64]
	vld1.32	{d0, d1, d2, d3}, [r4:128]
	vld1.32	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vld1.32 {d0, d1, d2, d3}, [r4]  @ encoding: [0x24,0xf9,0x8f,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vld1.32 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x24,0xf9,0x9f,0x02]
@ CHECK: vld1.32 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x24,0xf9,0xaf,0x02]
@ CHECK: vld1.32 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x24,0xf9,0xbf,0x02]

	vld1.32	{d0, d1, d2, d3}, [r4]!
	vld1.32	{d0, d1, d2, d3}, [r4:16]!
	vld1.32	{d0, d1, d2, d3}, [r4:32]!
	vld1.32	{d0, d1, d2, d3}, [r4:64]!
	vld1.32	{d0, d1, d2, d3}, [r4:128]!
	vld1.32	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vld1.32 {d0, d1, d2, d3}, [r4]! @ encoding: [0x24,0xf9,0x8d,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vld1.32 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x24,0xf9,0x9d,0x02]
@ CHECK: vld1.32 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x24,0xf9,0xad,0x02]
@ CHECK: vld1.32 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x24,0xf9,0xbd,0x02]

	vld1.32	{d0, d1, d2, d3}, [r4], r6
	vld1.32	{d0, d1, d2, d3}, [r4:16], r6
	vld1.32	{d0, d1, d2, d3}, [r4:32], r6
	vld1.32	{d0, d1, d2, d3}, [r4:64], r6
	vld1.32	{d0, d1, d2, d3}, [r4:128], r6
	vld1.32	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vld1.32 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x24,0xf9,0x86,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.32 {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vld1.32 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x24,0xf9,0x96,0x02]
@ CHECK: vld1.32 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x24,0xf9,0xa6,0x02]
@ CHECK: vld1.32 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x24,0xf9,0xb6,0x02]

	vld1.32	{d0[1]}, [r4]
	vld1.32	{d0[1]}, [r4:16]
	vld1.32	{d0[1]}, [r4:32]
	vld1.32	{d0[1]}, [r4:64]
	vld1.32	{d0[1]}, [r4:128]
	vld1.32	{d0[1]}, [r4:256]

@ CHECK: vld1.32 {d0[1]}, [r4]           @ encoding: [0xa4,0xf9,0x8f,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:16]
@ CHECK-ERRORS:                              ^
@ CHECK: vld1.32 {d0[1]}, [r4:32]        @ encoding: [0xa4,0xf9,0xbf,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:64]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:128]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:256]
@ CHECK-ERRORS:                              ^

	vld1.32	{d0[1]}, [r4]!
	vld1.32	{d0[1]}, [r4:16]!
	vld1.32	{d0[1]}, [r4:32]!
	vld1.32	{d0[1]}, [r4:64]!
	vld1.32	{d0[1]}, [r4:128]!
	vld1.32	{d0[1]}, [r4:256]!

@ CHECK: vld1.32 {d0[1]}, [r4]!          @ encoding: [0xa4,0xf9,0x8d,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:16]!
@ CHECK-ERRORS:                              ^
@ CHECK: vld1.32 {d0[1]}, [r4:32]!       @ encoding: [0xa4,0xf9,0xbd,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:64]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:128]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:256]!
@ CHECK-ERRORS:                              ^

	vld1.32	{d0[1]}, [r4], r6
	vld1.32	{d0[1]}, [r4:16], r6
	vld1.32	{d0[1]}, [r4:32], r6
	vld1.32	{d0[1]}, [r4:64], r6
	vld1.32	{d0[1]}, [r4:128], r6
	vld1.32	{d0[1]}, [r4:256], r6

@ CHECK: vld1.32 {d0[1]}, [r4], r6       @ encoding: [0xa4,0xf9,0x86,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:16], r6
@ CHECK-ERRORS:                              ^
@ CHECK: vld1.32 {d0[1]}, [r4:32], r6    @ encoding: [0xa4,0xf9,0xb6,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:64], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:128], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:256], r6
@ CHECK-ERRORS:                              ^

	vld1.32	{d0[]}, [r4]
	vld1.32	{d0[]}, [r4:16]
	vld1.32	{d0[]}, [r4:32]
	vld1.32	{d0[]}, [r4:64]
	vld1.32	{d0[]}, [r4:128]
	vld1.32	{d0[]}, [r4:256]

@ CHECK: vld1.32 {d0[]}, [r4]            @ encoding: [0xa4,0xf9,0x8f,0x0c]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[]}, [r4:16]
@ CHECK-ERRORS:                             ^
@ CHECK: vld1.32 {d0[]}, [r4:32]         @ encoding: [0xa4,0xf9,0x9f,0x0c]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[]}, [r4:64]
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[]}, [r4:128]
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[]}, [r4:256]
@ CHECK-ERRORS:                             ^

	vld1.32	{d0[]}, [r4]!
	vld1.32	{d0[]}, [r4:16]!
	vld1.32	{d0[]}, [r4:32]!
	vld1.32	{d0[]}, [r4:64]!
	vld1.32	{d0[]}, [r4:128]!
	vld1.32	{d0[]}, [r4:256]!

@ CHECK: vld1.32 {d0[]}, [r4]!           @ encoding: [0xa4,0xf9,0x8d,0x0c]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[]}, [r4:16]!
@ CHECK-ERRORS:                             ^
@ CHECK: vld1.32 {d0[]}, [r4:32]!        @ encoding: [0xa4,0xf9,0x9d,0x0c]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[]}, [r4:64]!
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[]}, [r4:128]!
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[]}, [r4:256]!
@ CHECK-ERRORS:                             ^

	vld1.32	{d0[]}, [r4], r6
	vld1.32	{d0[]}, [r4:16], r6
	vld1.32	{d0[]}, [r4:32], r6
	vld1.32	{d0[]}, [r4:64], r6
	vld1.32	{d0[]}, [r4:128], r6
	vld1.32	{d0[]}, [r4:256], r6

@ CHECK: vld1.32 {d0[]}, [r4], r6        @ encoding: [0xa4,0xf9,0x86,0x0c]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[]}, [r4:16], r6
@ CHECK-ERRORS:                             ^
@ CHECK: vld1.32 {d0[]}, [r4:32], r6     @ encoding: [0xa4,0xf9,0x96,0x0c]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[]}, [r4:64], r6
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[]}, [r4:128], r6
@ CHECK-ERRORS:                             ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[]}, [r4:256], r6
@ CHECK-ERRORS:                             ^

	vld1.32	{d0[], d1[]}, [r4]
	vld1.32	{d0[], d1[]}, [r4:16]
	vld1.32	{d0[], d1[]}, [r4:32]
	vld1.32	{d0[], d1[]}, [r4:64]
	vld1.32	{d0[], d1[]}, [r4:128]
	vld1.32	{d0[], d1[]}, [r4:256]

@ CHECK: vld1.32 {d0[], d1[]}, [r4]      @ encoding: [0xa4,0xf9,0xaf,0x0c]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[], d1[]}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.32 {d0[], d1[]}, [r4:32]   @ encoding: [0xa4,0xf9,0xbf,0x0c]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[], d1[]}, [r4:64]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[], d1[]}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[], d1[]}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld1.32	{d0[], d1[]}, [r4]!
	vld1.32	{d0[], d1[]}, [r4:16]!
	vld1.32	{d0[], d1[]}, [r4:32]!
	vld1.32	{d0[], d1[]}, [r4:64]!
	vld1.32	{d0[], d1[]}, [r4:128]!
	vld1.32	{d0[], d1[]}, [r4:256]!

@ CHECK: vld1.32 {d0[], d1[]}, [r4]!     @ encoding: [0xa4,0xf9,0xad,0x0c]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[], d1[]}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.32 {d0[], d1[]}, [r4:32]!  @ encoding: [0xa4,0xf9,0xbd,0x0c]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[], d1[]}, [r4:64]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[], d1[]}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[], d1[]}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld1.32	{d0[], d1[]}, [r4], r6
	vld1.32	{d0[], d1[]}, [r4:16], r6
	vld1.32	{d0[], d1[]}, [r4:32], r6
	vld1.32	{d0[], d1[]}, [r4:64], r6
	vld1.32	{d0[], d1[]}, [r4:128], r6
	vld1.32	{d0[], d1[]}, [r4:256], r6

@ CHECK: vld1.32 {d0[], d1[]}, [r4], r6  @ encoding: [0xa4,0xf9,0xa6,0x0c]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[], d1[]}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.32 {d0[], d1[]}, [r4:32], r6 @ encoding: [0xa4,0xf9,0xb6,0x0c]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[], d1[]}, [r4:64], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[], d1[]}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[], d1[]}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld1.32	{d0[1]}, [r4]
	vld1.32	{d0[1]}, [r4:16]
	vld1.32	{d0[1]}, [r4:32]
	vld1.32	{d0[1]}, [r4:64]
	vld1.32	{d0[1]}, [r4:128]
	vld1.32	{d0[1]}, [r4:256]

@ CHECK: vld1.32 {d0[1]}, [r4]           @ encoding: [0xa4,0xf9,0x8f,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:16]
@ CHECK-ERRORS:                              ^
@ CHECK: vld1.32 {d0[1]}, [r4:32]        @ encoding: [0xa4,0xf9,0xbf,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:64]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:128]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:256]
@ CHECK-ERRORS:                              ^

	vld1.32	{d0[1]}, [r4]!
	vld1.32	{d0[1]}, [r4:16]!
	vld1.32	{d0[1]}, [r4:32]!
	vld1.32	{d0[1]}, [r4:64]!
	vld1.32	{d0[1]}, [r4:128]!
	vld1.32	{d0[1]}, [r4:256]!

@ CHECK: vld1.32 {d0[1]}, [r4]!          @ encoding: [0xa4,0xf9,0x8d,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:16]!
@ CHECK-ERRORS:                              ^
@ CHECK: vld1.32 {d0[1]}, [r4:32]!       @ encoding: [0xa4,0xf9,0xbd,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:64]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:128]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:256]!
@ CHECK-ERRORS:                              ^

	vld1.32	{d0[1]}, [r4], r6
	vld1.32	{d0[1]}, [r4:16], r6
	vld1.32	{d0[1]}, [r4:32], r6
	vld1.32	{d0[1]}, [r4:64], r6
	vld1.32	{d0[1]}, [r4:128], r6
	vld1.32	{d0[1]}, [r4:256], r6

@ CHECK: vld1.32 {d0[1]}, [r4], r6       @ encoding: [0xa4,0xf9,0x86,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:16], r6
@ CHECK-ERRORS:                              ^
@ CHECK: vld1.32 {d0[1]}, [r4:32], r6    @ encoding: [0xa4,0xf9,0xb6,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:64], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:128], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld1.32 {d0[1]}, [r4:256], r6
@ CHECK-ERRORS:                              ^

	vld1.64	{d0}, [r4]
	vld1.64	{d0}, [r4:16]
	vld1.64	{d0}, [r4:32]
	vld1.64	{d0}, [r4:64]
	vld1.64	{d0}, [r4:128]
	vld1.64	{d0}, [r4:256]

@ CHECK: vld1.64 {d0}, [r4]              @ encoding: [0x24,0xf9,0xcf,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0}, [r4:16]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0}, [r4:32]
@ CHECK-ERRORS:                           ^
@ CHECK: vld1.64 {d0}, [r4:64]           @ encoding: [0x24,0xf9,0xdf,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0}, [r4:128]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0}, [r4:256]
@ CHECK-ERRORS:                           ^

	vld1.64	{d0}, [r4]!
	vld1.64	{d0}, [r4:16]!
	vld1.64	{d0}, [r4:32]!
	vld1.64	{d0}, [r4:64]!
	vld1.64	{d0}, [r4:128]!
	vld1.64	{d0}, [r4:256]!

@ CHECK: vld1.64 {d0}, [r4]!             @ encoding: [0x24,0xf9,0xcd,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0}, [r4:16]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0}, [r4:32]!
@ CHECK-ERRORS:                           ^
@ CHECK: vld1.64 {d0}, [r4:64]!          @ encoding: [0x24,0xf9,0xdd,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0}, [r4:128]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0}, [r4:256]!
@ CHECK-ERRORS:                           ^

	vld1.64	{d0}, [r4], r6
	vld1.64	{d0}, [r4:16], r6
	vld1.64	{d0}, [r4:32], r6
	vld1.64	{d0}, [r4:64], r6
	vld1.64	{d0}, [r4:128], r6
	vld1.64	{d0}, [r4:256], r6

@ CHECK: vld1.64 {d0}, [r4], r6          @ encoding: [0x24,0xf9,0xc6,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0}, [r4:16], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0}, [r4:32], r6
@ CHECK-ERRORS:                           ^
@ CHECK: vld1.64 {d0}, [r4:64], r6       @ encoding: [0x24,0xf9,0xd6,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0}, [r4:128], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0}, [r4:256], r6
@ CHECK-ERRORS:                           ^

	vld1.64	{d0, d1}, [r4]
	vld1.64	{d0, d1}, [r4:16]
	vld1.64	{d0, d1}, [r4:32]
	vld1.64	{d0, d1}, [r4:64]
	vld1.64	{d0, d1}, [r4:128]
	vld1.64	{d0, d1}, [r4:256]

@ CHECK: vld1.64 {d0, d1}, [r4]          @ encoding: [0x24,0xf9,0xcf,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vld1.64 {d0, d1}, [r4:64]       @ encoding: [0x24,0xf9,0xdf,0x0a]
@ CHECK: vld1.64 {d0, d1}, [r4:128]      @ encoding: [0x24,0xf9,0xef,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1}, [r4:256]
@ CHECK-ERRORS:                               ^

	vld1.64	{d0, d1}, [r4]!
	vld1.64	{d0, d1}, [r4:16]!
	vld1.64	{d0, d1}, [r4:32]!
	vld1.64	{d0, d1}, [r4:64]!
	vld1.64	{d0, d1}, [r4:128]!
	vld1.64	{d0, d1}, [r4:256]!

@ CHECK: vld1.64 {d0, d1}, [r4]!         @ encoding: [0x24,0xf9,0xcd,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vld1.64 {d0, d1}, [r4:64]!      @ encoding: [0x24,0xf9,0xdd,0x0a]
@ CHECK: vld1.64 {d0, d1}, [r4:128]!     @ encoding: [0x24,0xf9,0xed,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vld1.64	{d0, d1}, [r4], r6
	vld1.64	{d0, d1}, [r4:16], r6
	vld1.64	{d0, d1}, [r4:32], r6
	vld1.64	{d0, d1}, [r4:64], r6
	vld1.64	{d0, d1}, [r4:128], r6
	vld1.64	{d0, d1}, [r4:256], r6

@ CHECK: vld1.64 {d0, d1}, [r4], r6      @ encoding: [0x24,0xf9,0xc6,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vld1.64 {d0, d1}, [r4:64], r6   @ encoding: [0x24,0xf9,0xd6,0x0a]
@ CHECK: vld1.64 {d0, d1}, [r4:128], r6  @ encoding: [0x24,0xf9,0xe6,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vld1.64	{d0, d1, d2}, [r4]
	vld1.64	{d0, d1, d2}, [r4:16]
	vld1.64	{d0, d1, d2}, [r4:32]
	vld1.64	{d0, d1, d2}, [r4:64]
	vld1.64	{d0, d1, d2}, [r4:128]
	vld1.64	{d0, d1, d2}, [r4:256]

@ CHECK: vld1.64 {d0, d1, d2}, [r4]      @ encoding: [0x24,0xf9,0xcf,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.64 {d0, d1, d2}, [r4:64]   @ encoding: [0x24,0xf9,0xdf,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld1.64	{d0, d1, d2}, [r4]!
	vld1.64	{d0, d1, d2}, [r4:16]!
	vld1.64	{d0, d1, d2}, [r4:32]!
	vld1.64	{d0, d1, d2}, [r4:64]!
	vld1.64	{d0, d1, d2}, [r4:128]!
	vld1.64	{d0, d1, d2}, [r4:256]!

@ CHECK: vld1.64 {d0, d1, d2}, [r4]!     @ encoding: [0x24,0xf9,0xcd,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.64 {d0, d1, d2}, [r4:64]!  @ encoding: [0x24,0xf9,0xdd,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld1.64	{d0, d1, d2}, [r4], r6
	vld1.64	{d0, d1, d2}, [r4:16], r6
	vld1.64	{d0, d1, d2}, [r4:32], r6
	vld1.64	{d0, d1, d2}, [r4:64], r6
	vld1.64	{d0, d1, d2}, [r4:128], r6
	vld1.64	{d0, d1, d2}, [r4:256], r6

@ CHECK: vld1.64 {d0, d1, d2}, [r4], r6  @ encoding: [0x24,0xf9,0xc6,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld1.64 {d0, d1, d2}, [r4:64], r6 @ encoding: [0x24,0xf9,0xd6,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld1.64	{d0, d1, d2, d3}, [r4]
	vld1.64	{d0, d1, d2, d3}, [r4:16]
	vld1.64	{d0, d1, d2, d3}, [r4:32]
	vld1.64	{d0, d1, d2, d3}, [r4:64]
	vld1.64	{d0, d1, d2, d3}, [r4:128]
	vld1.64	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vld1.64 {d0, d1, d2, d3}, [r4]  @ encoding: [0x24,0xf9,0xcf,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vld1.64 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x24,0xf9,0xdf,0x02]
@ CHECK: vld1.64 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x24,0xf9,0xef,0x02]
@ CHECK: vld1.64 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x24,0xf9,0xff,0x02]

	vld1.64	{d0, d1, d2, d3}, [r4]!
	vld1.64	{d0, d1, d2, d3}, [r4:16]!
	vld1.64	{d0, d1, d2, d3}, [r4:32]!
	vld1.64	{d0, d1, d2, d3}, [r4:64]!
	vld1.64	{d0, d1, d2, d3}, [r4:128]!
	vld1.64	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vld1.64 {d0, d1, d2, d3}, [r4]! @ encoding: [0x24,0xf9,0xcd,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vld1.64 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x24,0xf9,0xdd,0x02]
@ CHECK: vld1.64 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x24,0xf9,0xed,0x02]
@ CHECK: vld1.64 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x24,0xf9,0xfd,0x02]

	vld1.64	{d0, d1, d2, d3}, [r4], r6
	vld1.64	{d0, d1, d2, d3}, [r4:16], r6
	vld1.64	{d0, d1, d2, d3}, [r4:32], r6
	vld1.64	{d0, d1, d2, d3}, [r4:64], r6
	vld1.64	{d0, d1, d2, d3}, [r4:128], r6
	vld1.64	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vld1.64 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x24,0xf9,0xc6,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld1.64 {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vld1.64 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x24,0xf9,0xd6,0x02]
@ CHECK: vld1.64 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x24,0xf9,0xe6,0x02]
@ CHECK: vld1.64 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x24,0xf9,0xf6,0x02]

	vld2.8	{d0, d1}, [r4]
	vld2.8	{d0, d1}, [r4:16]
	vld2.8	{d0, d1}, [r4:32]
	vld2.8	{d0, d1}, [r4:64]
	vld2.8	{d0, d1}, [r4:128]
	vld2.8	{d0, d1}, [r4:256]

@ CHECK: vld2.8 {d0, d1}, [r4]          @ encoding: [0x24,0xf9,0x0f,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.8 {d0, d1}, [r4:64]       @ encoding: [0x24,0xf9,0x1f,0x08]
@ CHECK: vld2.8 {d0, d1}, [r4:128]      @ encoding: [0x24,0xf9,0x2f,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1}, [r4:256]
@ CHECK-ERRORS:                               ^

	vld2.8	{d0, d1}, [r4]!
	vld2.8	{d0, d1}, [r4:16]!
	vld2.8	{d0, d1}, [r4:32]!
	vld2.8	{d0, d1}, [r4:64]!
	vld2.8	{d0, d1}, [r4:128]!
	vld2.8	{d0, d1}, [r4:256]!

@ CHECK: vld2.8 {d0, d1}, [r4]!         @ encoding: [0x24,0xf9,0x0d,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.8 {d0, d1}, [r4:64]!      @ encoding: [0x24,0xf9,0x1d,0x08]
@ CHECK: vld2.8 {d0, d1}, [r4:128]!     @ encoding: [0x24,0xf9,0x2d,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vld2.8	{d0, d1}, [r4], r6
	vld2.8	{d0, d1}, [r4:16], r6
	vld2.8	{d0, d1}, [r4:32], r6
	vld2.8	{d0, d1}, [r4:64], r6
	vld2.8	{d0, d1}, [r4:128], r6
	vld2.8	{d0, d1}, [r4:256], r6

@ CHECK: vld2.8 {d0, d1}, [r4], r6      @ encoding: [0x24,0xf9,0x06,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.8 {d0, d1}, [r4:64], r6   @ encoding: [0x24,0xf9,0x16,0x08]
@ CHECK: vld2.8 {d0, d1}, [r4:128], r6  @ encoding: [0x24,0xf9,0x26,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vld2.8	{d0, d2}, [r4]
	vld2.8	{d0, d2}, [r4:16]
	vld2.8	{d0, d2}, [r4:32]
	vld2.8	{d0, d2}, [r4:64]
	vld2.8	{d0, d2}, [r4:128]
	vld2.8	{d0, d2}, [r4:256]

@ CHECK: vld2.8 {d0, d2}, [r4]          @ encoding: [0x24,0xf9,0x0f,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d2}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d2}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.8 {d0, d2}, [r4:64]       @ encoding: [0x24,0xf9,0x1f,0x09]
@ CHECK: vld2.8 {d0, d2}, [r4:128]      @ encoding: [0x24,0xf9,0x2f,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d2}, [r4:256]
@ CHECK-ERRORS:                               ^

	vld2.8	{d0, d2}, [r4]!
	vld2.8	{d0, d2}, [r4:16]!
	vld2.8	{d0, d2}, [r4:32]!
	vld2.8	{d0, d2}, [r4:64]!
	vld2.8	{d0, d2}, [r4:128]!
	vld2.8	{d0, d2}, [r4:256]!

@ CHECK: vld2.8 {d0, d2}, [r4]!         @ encoding: [0x24,0xf9,0x0d,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d2}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d2}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.8 {d0, d2}, [r4:64]!      @ encoding: [0x24,0xf9,0x1d,0x09]
@ CHECK: vld2.8 {d0, d2}, [r4:128]!     @ encoding: [0x24,0xf9,0x2d,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d2}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vld2.8	{d0, d2}, [r4], r6
	vld2.8	{d0, d2}, [r4:16], r6
	vld2.8	{d0, d2}, [r4:32], r6
	vld2.8	{d0, d2}, [r4:64], r6
	vld2.8	{d0, d2}, [r4:128], r6
	vld2.8	{d0, d2}, [r4:256], r6

@ CHECK: vld2.8 {d0, d2}, [r4], r6      @ encoding: [0x24,0xf9,0x06,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d2}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d2}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.8 {d0, d2}, [r4:64], r6   @ encoding: [0x24,0xf9,0x16,0x09]
@ CHECK: vld2.8 {d0, d2}, [r4:128], r6  @ encoding: [0x24,0xf9,0x26,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d2}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vld2.8	{d0, d1, d2, d3}, [r4]
	vld2.8	{d0, d1, d2, d3}, [r4:16]
	vld2.8	{d0, d1, d2, d3}, [r4:32]
	vld2.8	{d0, d1, d2, d3}, [r4:64]
	vld2.8	{d0, d1, d2, d3}, [r4:128]
	vld2.8	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vld2.8 {d0, d1, d2, d3}, [r4]  @ encoding: [0x24,0xf9,0x0f,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vld2.8 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x24,0xf9,0x1f,0x03]
@ CHECK: vld2.8 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x24,0xf9,0x2f,0x03]
@ CHECK: vld2.8 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x24,0xf9,0x3f,0x03]

	vld2.8	{d0, d1, d2, d3}, [r4]!
	vld2.8	{d0, d1, d2, d3}, [r4:16]!
	vld2.8	{d0, d1, d2, d3}, [r4:32]!
	vld2.8	{d0, d1, d2, d3}, [r4:64]!
	vld2.8	{d0, d1, d2, d3}, [r4:128]!
	vld2.8	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vld2.8 {d0, d1, d2, d3}, [r4]! @ encoding: [0x24,0xf9,0x0d,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vld2.8 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x24,0xf9,0x1d,0x03]
@ CHECK: vld2.8 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x24,0xf9,0x2d,0x03]
@ CHECK: vld2.8 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x24,0xf9,0x3d,0x03]

	vld2.8	{d0, d1, d2, d3}, [r4], r6
	vld2.8	{d0, d1, d2, d3}, [r4:16], r6
	vld2.8	{d0, d1, d2, d3}, [r4:32], r6
	vld2.8	{d0, d1, d2, d3}, [r4:64], r6
	vld2.8	{d0, d1, d2, d3}, [r4:128], r6
	vld2.8	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vld2.8 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x24,0xf9,0x06,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.8  {d0, d1, d2, d3}, [r4:32], r6
@ CHECK: vld2.8 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x24,0xf9,0x16,0x03]
@ CHECK: vld2.8 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x24,0xf9,0x26,0x03]
@ CHECK: vld2.8 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x24,0xf9,0x36,0x03]

	vld2.8	{d0[2], d1[2]}, [r4]
	vld2.8	{d0[2], d1[2]}, [r4:16]
	vld2.8	{d0[2], d1[2]}, [r4:32]
	vld2.8	{d0[2], d1[2]}, [r4:64]
	vld2.8	{d0[2], d1[2]}, [r4:128]
	vld2.8	{d0[2], d1[2]}, [r4:256]

@ CHECK: vld2.8 {d0[2], d1[2]}, [r4]    @ encoding: [0xa4,0xf9,0x4f,0x01]
@ CHECK: vld2.8 {d0[2], d1[2]}, [r4:16] @ encoding: [0xa4,0xf9,0x5f,0x01]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[2], d1[2]}, [r4:32]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[2], d1[2]}, [r4:64]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[2], d1[2]}, [r4:128]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[2], d1[2]}, [r4:256]
@ CHECK-ERRORS:                                     ^

	vld2.8	{d0[2], d1[2]}, [r4]!
	vld2.8	{d0[2], d1[2]}, [r4:16]!
	vld2.8	{d0[2], d1[2]}, [r4:32]!
	vld2.8	{d0[2], d1[2]}, [r4:64]!
	vld2.8	{d0[2], d1[2]}, [r4:128]!
	vld2.8	{d0[2], d1[2]}, [r4:256]!

@ CHECK: vld2.8 {d0[2], d1[2]}, [r4]!   @ encoding: [0xa4,0xf9,0x4d,0x01]
@ CHECK: vld2.8 {d0[2], d1[2]}, [r4:16]! @ encoding: [0xa4,0xf9,0x5d,0x01]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[2], d1[2]}, [r4:32]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[2], d1[2]}, [r4:64]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[2], d1[2]}, [r4:128]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[2], d1[2]}, [r4:256]!
@ CHECK-ERRORS:                                     ^

	vld2.8	{d0[2], d1[2]}, [r4], r6
	vld2.8	{d0[2], d1[2]}, [r4:16], r6
	vld2.8	{d0[2], d1[2]}, [r4:32], r6
	vld2.8	{d0[2], d1[2]}, [r4:64], r6
	vld2.8	{d0[2], d1[2]}, [r4:128], r6
	vld2.8	{d0[2], d1[2]}, [r4:256], r6

@ CHECK: vld2.8 {d0[2], d1[2]}, [r4], r6 @ encoding: [0xa4,0xf9,0x46,0x01]
@ CHECK: vld2.8 {d0[2], d1[2]}, [r4:16], r6 @ encoding: [0xa4,0xf9,0x56,0x01]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[2], d1[2]}, [r4:32], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[2], d1[2]}, [r4:64], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[2], d1[2]}, [r4:128], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[2], d1[2]}, [r4:256], r6
@ CHECK-ERRORS:                                     ^

	vld2.8	{d0[], d1[]}, [r4]
	vld2.8	{d0[], d1[]}, [r4:16]
	vld2.8	{d0[], d1[]}, [r4:32]
	vld2.8	{d0[], d1[]}, [r4:64]
	vld2.8	{d0[], d1[]}, [r4:128]
	vld2.8	{d0[], d1[]}, [r4:256]

@ CHECK: vld2.8 {d0[], d1[]}, [r4]      @ encoding: [0xa4,0xf9,0x0f,0x0d]
@ CHECK: vld2.8 {d0[], d1[]}, [r4:16]   @ encoding: [0xa4,0xf9,0x1f,0x0d]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d1[]}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d1[]}, [r4:64]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d1[]}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d1[]}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld2.8	{d0[], d1[]}, [r4]!
	vld2.8	{d0[], d1[]}, [r4:16]!
	vld2.8	{d0[], d1[]}, [r4:32]!
	vld2.8	{d0[], d1[]}, [r4:64]!
	vld2.8	{d0[], d1[]}, [r4:128]!
	vld2.8	{d0[], d1[]}, [r4:256]!

@ CHECK: vld2.8 {d0[], d1[]}, [r4]!     @ encoding: [0xa4,0xf9,0x0d,0x0d]
@ CHECK: vld2.8 {d0[], d1[]}, [r4:16]!  @ encoding: [0xa4,0xf9,0x1d,0x0d]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d1[]}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d1[]}, [r4:64]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d1[]}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d1[]}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld2.8	{d0[], d1[]}, [r4], r6
	vld2.8	{d0[], d1[]}, [r4:16], r6
	vld2.8	{d0[], d1[]}, [r4:32], r6
	vld2.8	{d0[], d1[]}, [r4:64], r6
	vld2.8	{d0[], d1[]}, [r4:128], r6
	vld2.8	{d0[], d1[]}, [r4:256], r6

@ CHECK: vld2.8 {d0[], d1[]}, [r4], r6  @ encoding: [0xa4,0xf9,0x06,0x0d]
@ CHECK: vld2.8 {d0[], d1[]}, [r4:16], r6 @ encoding: [0xa4,0xf9,0x16,0x0d]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d1[]}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d1[]}, [r4:64], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d1[]}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d1[]}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld2.8	{d0[], d2[]}, [r4]
	vld2.8	{d0[], d2[]}, [r4:16]
	vld2.8	{d0[], d2[]}, [r4:32]
	vld2.8	{d0[], d2[]}, [r4:64]
	vld2.8	{d0[], d2[]}, [r4:128]
	vld2.8	{d0[], d2[]}, [r4:256]

@ CHECK: vld2.8 {d0[], d2[]}, [r4]      @ encoding: [0xa4,0xf9,0x2f,0x0d]
@ CHECK: vld2.8 {d0[], d2[]}, [r4:16]   @ encoding: [0xa4,0xf9,0x3f,0x0d]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d2[]}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d2[]}, [r4:64]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d2[]}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d2[]}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld2.8	{d0[], d2[]}, [r4]!
	vld2.8	{d0[], d2[]}, [r4:16]!
	vld2.8	{d0[], d2[]}, [r4:32]!
	vld2.8	{d0[], d2[]}, [r4:64]!
	vld2.8	{d0[], d2[]}, [r4:128]!
	vld2.8	{d0[], d2[]}, [r4:256]!

@ CHECK: vld2.8 {d0[], d2[]}, [r4]!     @ encoding: [0xa4,0xf9,0x2d,0x0d]
@ CHECK: vld2.8 {d0[], d2[]}, [r4:16]!  @ encoding: [0xa4,0xf9,0x3d,0x0d]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d2[]}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d2[]}, [r4:64]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d2[]}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d2[]}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld2.8	{d0[], d2[]}, [r4], r6
	vld2.8	{d0[], d2[]}, [r4:16], r6
	vld2.8	{d0[], d2[]}, [r4:32], r6
	vld2.8	{d0[], d2[]}, [r4:64], r6
	vld2.8	{d0[], d2[]}, [r4:128], r6
	vld2.8	{d0[], d2[]}, [r4:256], r6

@ CHECK: vld2.8 {d0[], d2[]}, [r4], r6  @ encoding: [0xa4,0xf9,0x26,0x0d]
@ CHECK: vld2.8 {d0[], d2[]}, [r4:16], r6 @ encoding: [0xa4,0xf9,0x36,0x0d]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d2[]}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d2[]}, [r4:64], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d2[]}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vld2.8  {d0[], d2[]}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld2.16	{d0, d1}, [r4]
	vld2.16	{d0, d1}, [r4:16]
	vld2.16	{d0, d1}, [r4:32]
	vld2.16	{d0, d1}, [r4:64]
	vld2.16	{d0, d1}, [r4:128]
	vld2.16	{d0, d1}, [r4:256]

@ CHECK: vld2.16 {d0, d1}, [r4]          @ encoding: [0x24,0xf9,0x4f,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.16 {d0, d1}, [r4:64]       @ encoding: [0x24,0xf9,0x5f,0x08]
@ CHECK: vld2.16 {d0, d1}, [r4:128]      @ encoding: [0x24,0xf9,0x6f,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1}, [r4:256]
@ CHECK-ERRORS:                               ^

	vld2.16	{d0, d1}, [r4]!
	vld2.16	{d0, d1}, [r4:16]!
	vld2.16	{d0, d1}, [r4:32]!
	vld2.16	{d0, d1}, [r4:64]!
	vld2.16	{d0, d1}, [r4:128]!
	vld2.16	{d0, d1}, [r4:256]!

@ CHECK: vld2.16 {d0, d1}, [r4]!         @ encoding: [0x24,0xf9,0x4d,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.16 {d0, d1}, [r4:64]!      @ encoding: [0x24,0xf9,0x5d,0x08]
@ CHECK: vld2.16 {d0, d1}, [r4:128]!     @ encoding: [0x24,0xf9,0x6d,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vld2.16	{d0, d1}, [r4], r6
	vld2.16	{d0, d1}, [r4:16], r6
	vld2.16	{d0, d1}, [r4:32], r6
	vld2.16	{d0, d1}, [r4:64], r6
	vld2.16	{d0, d1}, [r4:128], r6
	vld2.16	{d0, d1}, [r4:256], r6

@ CHECK: vld2.16 {d0, d1}, [r4], r6      @ encoding: [0x24,0xf9,0x46,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.16 {d0, d1}, [r4:64], r6   @ encoding: [0x24,0xf9,0x56,0x08]
@ CHECK: vld2.16 {d0, d1}, [r4:128], r6  @ encoding: [0x24,0xf9,0x66,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vld2.16	{d0, d2}, [r4]
	vld2.16	{d0, d2}, [r4:16]
	vld2.16	{d0, d2}, [r4:32]
	vld2.16	{d0, d2}, [r4:64]
	vld2.16	{d0, d2}, [r4:128]
	vld2.16	{d0, d2}, [r4:256]

@ CHECK: vld2.16 {d0, d2}, [r4]          @ encoding: [0x24,0xf9,0x4f,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d2}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d2}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.16 {d0, d2}, [r4:64]       @ encoding: [0x24,0xf9,0x5f,0x09]
@ CHECK: vld2.16 {d0, d2}, [r4:128]      @ encoding: [0x24,0xf9,0x6f,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d2}, [r4:256]
@ CHECK-ERRORS:                               ^

	vld2.16	{d0, d2}, [r4]!
	vld2.16	{d0, d2}, [r4:16]!
	vld2.16	{d0, d2}, [r4:32]!
	vld2.16	{d0, d2}, [r4:64]!
	vld2.16	{d0, d2}, [r4:128]!
	vld2.16	{d0, d2}, [r4:256]!

@ CHECK: vld2.16 {d0, d2}, [r4]!         @ encoding: [0x24,0xf9,0x4d,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d2}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d2}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.16 {d0, d2}, [r4:64]!      @ encoding: [0x24,0xf9,0x5d,0x09]
@ CHECK: vld2.16 {d0, d2}, [r4:128]!     @ encoding: [0x24,0xf9,0x6d,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d2}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vld2.16	{d0, d2}, [r4], r6
	vld2.16	{d0, d2}, [r4:16], r6
	vld2.16	{d0, d2}, [r4:32], r6
	vld2.16	{d0, d2}, [r4:64], r6
	vld2.16	{d0, d2}, [r4:128], r6
	vld2.16	{d0, d2}, [r4:256], r6

@ CHECK: vld2.16 {d0, d2}, [r4], r6      @ encoding: [0x24,0xf9,0x46,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d2}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d2}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.16 {d0, d2}, [r4:64], r6   @ encoding: [0x24,0xf9,0x56,0x09]
@ CHECK: vld2.16 {d0, d2}, [r4:128], r6  @ encoding: [0x24,0xf9,0x66,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d2}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vld2.16	{d0, d1, d2, d3}, [r4]
	vld2.16	{d0, d1, d2, d3}, [r4:16]
	vld2.16	{d0, d1, d2, d3}, [r4:32]
	vld2.16	{d0, d1, d2, d3}, [r4:64]
	vld2.16	{d0, d1, d2, d3}, [r4:128]
	vld2.16	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vld2.16 {d0, d1, d2, d3}, [r4]  @ encoding: [0x24,0xf9,0x4f,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vld2.16 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x24,0xf9,0x5f,0x03]
@ CHECK: vld2.16 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x24,0xf9,0x6f,0x03]
@ CHECK: vld2.16 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x24,0xf9,0x7f,0x03]

	vld2.16	{d0, d1, d2, d3}, [r4]!
	vld2.16	{d0, d1, d2, d3}, [r4:16]!
	vld2.16	{d0, d1, d2, d3}, [r4:32]!
	vld2.16	{d0, d1, d2, d3}, [r4:64]!
	vld2.16	{d0, d1, d2, d3}, [r4:128]!
	vld2.16	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vld2.16 {d0, d1, d2, d3}, [r4]! @ encoding: [0x24,0xf9,0x4d,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vld2.16 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x24,0xf9,0x5d,0x03]
@ CHECK: vld2.16 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x24,0xf9,0x6d,0x03]
@ CHECK: vld2.16 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x24,0xf9,0x7d,0x03]

	vld2.16	{d0, d1, d2, d3}, [r4], r6
	vld2.16	{d0, d1, d2, d3}, [r4:16], r6
	vld2.16	{d0, d1, d2, d3}, [r4:32], r6
	vld2.16	{d0, d1, d2, d3}, [r4:64], r6
	vld2.16	{d0, d1, d2, d3}, [r4:128], r6
	vld2.16	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vld2.16 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x24,0xf9,0x46,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.16 {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vld2.16 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x24,0xf9,0x56,0x03]
@ CHECK: vld2.16 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x24,0xf9,0x66,0x03]
@ CHECK: vld2.16 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x24,0xf9,0x76,0x03]

	vld2.16	{d0[2], d1[2]}, [r4]
	vld2.16	{d0[2], d1[2]}, [r4:16]
	vld2.16	{d0[2], d1[2]}, [r4:32]
	vld2.16	{d0[2], d1[2]}, [r4:64]
	vld2.16	{d0[2], d1[2]}, [r4:128]
	vld2.16	{d0[2], d1[2]}, [r4:256]

@ CHECK: vld2.16 {d0[2], d1[2]}, [r4]    @ encoding: [0xa4,0xf9,0x8f,0x05]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d1[2]}, [r4:16]
@ CHECK-ERRORS:                                     ^
@ CHECK: vld2.16 {d0[2], d1[2]}, [r4:32] @ encoding: [0xa4,0xf9,0x9f,0x05]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d1[2]}, [r4:64]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d1[2]}, [r4:128]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d1[2]}, [r4:256]
@ CHECK-ERRORS:                                     ^

	vld2.16	{d0[2], d1[2]}, [r4]!
	vld2.16	{d0[2], d1[2]}, [r4:16]!
	vld2.16	{d0[2], d1[2]}, [r4:32]!
	vld2.16	{d0[2], d1[2]}, [r4:64]!
	vld2.16	{d0[2], d1[2]}, [r4:128]!
	vld2.16	{d0[2], d1[2]}, [r4:256]!

@ CHECK: vld2.16 {d0[2], d1[2]}, [r4]!   @ encoding: [0xa4,0xf9,0x8d,0x05]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d1[2]}, [r4:16]!
@ CHECK-ERRORS:                                     ^
@ CHECK: vld2.16 {d0[2], d1[2]}, [r4:32]! @ encoding: [0xa4,0xf9,0x9d,0x05]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d1[2]}, [r4:64]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d1[2]}, [r4:128]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d1[2]}, [r4:256]!
@ CHECK-ERRORS:                                     ^

	vld2.16	{d0[2], d1[2]}, [r4], r6
	vld2.16	{d0[2], d1[2]}, [r4:16], r6
	vld2.16	{d0[2], d1[2]}, [r4:32], r6
	vld2.16	{d0[2], d1[2]}, [r4:64], r6
	vld2.16	{d0[2], d1[2]}, [r4:128], r6
	vld2.16	{d0[2], d1[2]}, [r4:256], r6

@ CHECK: vld2.16 {d0[2], d1[2]}, [r4], r6 @ encoding: [0xa4,0xf9,0x86,0x05]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d1[2]}, [r4:16], r6
@ CHECK-ERRORS:                                     ^
@ CHECK: vld2.16 {d0[2], d1[2]}, [r4:32], r6 @ encoding: [0xa4,0xf9,0x96,0x05]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d1[2]}, [r4:64], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d1[2]}, [r4:128], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d1[2]}, [r4:256], r6
@ CHECK-ERRORS:                                     ^

	vld2.16	{d0[2], d2[2]}, [r4]
	vld2.16	{d0[2], d2[2]}, [r4:16]
	vld2.16	{d0[2], d2[2]}, [r4:32]
	vld2.16	{d0[2], d2[2]}, [r4:64]
	vld2.16	{d0[2], d2[2]}, [r4:128]
	vld2.16	{d0[2], d2[2]}, [r4:256]

@ CHECK: vld2.16 {d0[2], d2[2]}, [r4]    @ encoding: [0xa4,0xf9,0xaf,0x05]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d2[2]}, [r4:16]
@ CHECK-ERRORS:                                     ^
@ CHECK: vld2.16 {d0[2], d2[2]}, [r4:32] @ encoding: [0xa4,0xf9,0xbf,0x05]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d2[2]}, [r4:64]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d2[2]}, [r4:128]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d2[2]}, [r4:256]
@ CHECK-ERRORS:                                     ^

	vld2.16	{d0[2], d2[2]}, [r4]!
	vld2.16	{d0[2], d2[2]}, [r4:16]!
	vld2.16	{d0[2], d2[2]}, [r4:32]!
	vld2.16	{d0[2], d2[2]}, [r4:64]!
	vld2.16	{d0[2], d2[2]}, [r4:128]!
	vld2.16	{d0[2], d2[2]}, [r4:256]!

@ CHECK: vld2.16 {d0[2], d1[2]}, [r4]!   @ encoding: [0xa4,0xf9,0xad,0x05]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d2[2]}, [r4:16]!
@ CHECK-ERRORS:                                     ^
@ CHECK: vld2.16 {d0[2], d1[2]}, [r4:32]! @ encoding: [0xa4,0xf9,0xbd,0x05]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d2[2]}, [r4:64]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d2[2]}, [r4:128]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d2[2]}, [r4:256]!
@ CHECK-ERRORS:                                     ^

	vld2.16	{d0[2], d2[2]}, [r4], r6
	vld2.16	{d0[2], d2[2]}, [r4:16], r6
	vld2.16	{d0[2], d2[2]}, [r4:32], r6
	vld2.16	{d0[2], d2[2]}, [r4:64], r6
	vld2.16	{d0[2], d2[2]}, [r4:128], r6
	vld2.16	{d0[2], d2[2]}, [r4:256], r6

@ CHECK: vld2.16 {d0[2], d2[2]}, [r4], r6 @ encoding: [0xa4,0xf9,0xa6,0x05]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d2[2]}, [r4:16], r6
@ CHECK-ERRORS:                                     ^
@ CHECK: vld2.16 {d0[2], d2[2]}, [r4:32], r6 @ encoding: [0xa4,0xf9,0xb6,0x05]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d2[2]}, [r4:64], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d2[2]}, [r4:128], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[2], d2[2]}, [r4:256], r6
@ CHECK-ERRORS:                                     ^

	vld2.16	{d0[], d1[]}, [r4]
	vld2.16	{d0[], d1[]}, [r4:16]
	vld2.16	{d0[], d1[]}, [r4:32]
	vld2.16	{d0[], d1[]}, [r4:64]
	vld2.16	{d0[], d1[]}, [r4:128]
	vld2.16	{d0[], d1[]}, [r4:256]

@ CHECK: vld2.16 {d0[], d1[]}, [r4]      @ encoding: [0xa4,0xf9,0x4f,0x0d]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d1[]}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld2.16 {d0[], d1[]}, [r4:32]   @ encoding: [0xa4,0xf9,0x5f,0x0d]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d1[]}, [r4:64]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d1[]}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d1[]}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld2.16	{d0[], d1[]}, [r4]!
	vld2.16	{d0[], d1[]}, [r4:16]!
	vld2.16	{d0[], d1[]}, [r4:32]!
	vld2.16	{d0[], d1[]}, [r4:64]!
	vld2.16	{d0[], d1[]}, [r4:128]!
	vld2.16	{d0[], d1[]}, [r4:256]!

@ CHECK: vld2.16 {d0[], d1[]}, [r4]!     @ encoding: [0xa4,0xf9,0x4d,0x0d]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d1[]}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld2.16 {d0[], d1[]}, [r4:32]!  @ encoding: [0xa4,0xf9,0x5d,0x0d]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d1[]}, [r4:64]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d1[]}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d1[]}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld2.16	{d0[], d1[]}, [r4], r6
	vld2.16	{d0[], d1[]}, [r4:16], r6
	vld2.16	{d0[], d1[]}, [r4:32], r6
	vld2.16	{d0[], d1[]}, [r4:64], r6
	vld2.16	{d0[], d1[]}, [r4:128], r6
	vld2.16	{d0[], d1[]}, [r4:256], r6

@ CHECK: vld2.16 {d0[], d1[]}, [r4], r6  @ encoding: [0xa4,0xf9,0x46,0x0d]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d1[]}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld2.16 {d0[], d1[]}, [r4:32], r6 @ encoding: [0xa4,0xf9,0x56,0x0d]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d1[]}, [r4:64], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d1[]}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d1[]}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld2.16	{d0[], d2[]}, [r4]
	vld2.16	{d0[], d2[]}, [r4:16]
	vld2.16	{d0[], d2[]}, [r4:32]
	vld2.16	{d0[], d2[]}, [r4:64]
	vld2.16	{d0[], d2[]}, [r4:128]
	vld2.16	{d0[], d2[]}, [r4:256]

@ CHECK: vld2.16 {d0[], d2[]}, [r4]      @ encoding: [0xa4,0xf9,0x6f,0x0d]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d2[]}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld2.16 {d0[], d2[]}, [r4:32]   @ encoding: [0xa4,0xf9,0x7f,0x0d]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d2[]}, [r4:64]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d2[]}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d2[]}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld2.16	{d0[], d2[]}, [r4]!
	vld2.16	{d0[], d2[]}, [r4:16]!
	vld2.16	{d0[], d2[]}, [r4:32]!
	vld2.16	{d0[], d2[]}, [r4:64]!
	vld2.16	{d0[], d2[]}, [r4:128]!
	vld2.16	{d0[], d2[]}, [r4:256]!

@ CHECK: vld2.16 {d0[], d2[]}, [r4]!     @ encoding: [0xa4,0xf9,0x6d,0x0d]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d2[]}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld2.16 {d0[], d2[]}, [r4:32]!  @ encoding: [0xa4,0xf9,0x7d,0x0d]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d2[]}, [r4:64]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d2[]}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d2[]}, [r4:256]!

	vld2.16	{d0[], d2[]}, [r4], r6
	vld2.16	{d0[], d2[]}, [r4:16], r6
	vld2.16	{d0[], d2[]}, [r4:32], r6
	vld2.16	{d0[], d2[]}, [r4:64], r6
	vld2.16	{d0[], d2[]}, [r4:128], r6
	vld2.16	{d0[], d2[]}, [r4:256], r6

@ CHECK: vld2.16 {d0[], d2[]}, [r4], r6  @ encoding: [0xa4,0xf9,0x66,0x0d]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d2[]}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld2.16 {d0[], d2[]}, [r4:32], r6 @ encoding: [0xa4,0xf9,0x76,0x0d]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d2[]}, [r4:64], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d2[]}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld2.16 {d0[], d2[]}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld2.32	{d0, d1}, [r4]
	vld2.32	{d0, d1}, [r4:16]
	vld2.32	{d0, d1}, [r4:32]
	vld2.32	{d0, d1}, [r4:64]
	vld2.32	{d0, d1}, [r4:128]
	vld2.32	{d0, d1}, [r4:256]

@ CHECK: vld2.32 {d0, d1}, [r4]          @ encoding: [0x24,0xf9,0x8f,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.32 {d0, d1}, [r4:64]       @ encoding: [0x24,0xf9,0x9f,0x08]
@ CHECK: vld2.32 {d0, d1}, [r4:128]      @ encoding: [0x24,0xf9,0xaf,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1}, [r4:256]
@ CHECK-ERRORS:                               ^

	vld2.32	{d0, d1}, [r4]!
	vld2.32	{d0, d1}, [r4:16]!
	vld2.32	{d0, d1}, [r4:32]!
	vld2.32	{d0, d1}, [r4:64]!
	vld2.32	{d0, d1}, [r4:128]!
	vld2.32	{d0, d1}, [r4:256]!

@ CHECK: vld2.32 {d0, d1}, [r4]!         @ encoding: [0x24,0xf9,0x8d,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.32 {d0, d1}, [r4:64]!      @ encoding: [0x24,0xf9,0x9d,0x08]
@ CHECK: vld2.32 {d0, d1}, [r4:128]!     @ encoding: [0x24,0xf9,0xad,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vld2.32	{d0, d1}, [r4], r6
	vld2.32	{d0, d1}, [r4:16], r6
	vld2.32	{d0, d1}, [r4:32], r6
	vld2.32	{d0, d1}, [r4:64], r6
	vld2.32	{d0, d1}, [r4:128], r6
	vld2.32	{d0, d1}, [r4:256], r6

@ CHECK: vld2.32 {d0, d1}, [r4], r6      @ encoding: [0x24,0xf9,0x86,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.32 {d0, d1}, [r4:64], r6   @ encoding: [0x24,0xf9,0x96,0x08]
@ CHECK: vld2.32 {d0, d1}, [r4:128], r6  @ encoding: [0x24,0xf9,0xa6,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vld2.32	{d0, d2}, [r4]
	vld2.32	{d0, d2}, [r4:16]
	vld2.32	{d0, d2}, [r4:32]
	vld2.32	{d0, d2}, [r4:64]
	vld2.32	{d0, d2}, [r4:128]
	vld2.32	{d0, d2}, [r4:256]

@ CHECK: vld2.32 {d0, d2}, [r4]          @ encoding: [0x24,0xf9,0x8f,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d2}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d2}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.32 {d0, d2}, [r4:64]       @ encoding: [0x24,0xf9,0x9f,0x09]
@ CHECK: vld2.32 {d0, d2}, [r4:128]      @ encoding: [0x24,0xf9,0xaf,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d2}, [r4:256]
@ CHECK-ERRORS:                               ^

	vld2.32	{d0, d2}, [r4]!
	vld2.32	{d0, d2}, [r4:16]!
	vld2.32	{d0, d2}, [r4:32]!
	vld2.32	{d0, d2}, [r4:64]!
	vld2.32	{d0, d2}, [r4:128]!
	vld2.32	{d0, d2}, [r4:256]!

@ CHECK: vld2.32 {d0, d2}, [r4]!         @ encoding: [0x24,0xf9,0x8d,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d2}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d2}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.32 {d0, d2}, [r4:64]!      @ encoding: [0x24,0xf9,0x9d,0x09]
@ CHECK: vld2.32 {d0, d2}, [r4:128]!     @ encoding: [0x24,0xf9,0xad,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d2}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vld2.32	{d0, d2}, [r4], r6
	vld2.32	{d0, d2}, [r4:16], r6
	vld2.32	{d0, d2}, [r4:32], r6
	vld2.32	{d0, d2}, [r4:64], r6
	vld2.32	{d0, d2}, [r4:128], r6
	vld2.32	{d0, d2}, [r4:256], r6

@ CHECK: vld2.32 {d0, d2}, [r4], r6      @ encoding: [0x24,0xf9,0x86,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d2}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d2}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vld2.32 {d0, d2}, [r4:64], r6   @ encoding: [0x24,0xf9,0x96,0x09]
@ CHECK: vld2.32 {d0, d2}, [r4:128], r6  @ encoding: [0x24,0xf9,0xa6,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d2}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vld2.32	{d0, d1, d2, d3}, [r4]
	vld2.32	{d0, d1, d2, d3}, [r4:16]
	vld2.32	{d0, d1, d2, d3}, [r4:32]
	vld2.32	{d0, d1, d2, d3}, [r4:64]
	vld2.32	{d0, d1, d2, d3}, [r4:128]
	vld2.32	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vld2.32 {d0, d1, d2, d3}, [r4]  @ encoding: [0x24,0xf9,0x8f,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vld2.32 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x24,0xf9,0x9f,0x03]
@ CHECK: vld2.32 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x24,0xf9,0xaf,0x03]
@ CHECK: vld2.32 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x24,0xf9,0xbf,0x03]

	vld2.32	{d0, d1, d2, d3}, [r4]!
	vld2.32	{d0, d1, d2, d3}, [r4:16]!
	vld2.32	{d0, d1, d2, d3}, [r4:32]!
	vld2.32	{d0, d1, d2, d3}, [r4:64]!
	vld2.32	{d0, d1, d2, d3}, [r4:128]!
	vld2.32	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vld2.32 {d0, d1, d2, d3}, [r4]! @ encoding: [0x24,0xf9,0x8d,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vld2.32 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x24,0xf9,0x9d,0x03]
@ CHECK: vld2.32 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x24,0xf9,0xad,0x03]
@ CHECK: vld2.32 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x24,0xf9,0xbd,0x03]

	vld2.32	{d0, d1, d2, d3}, [r4], r6
	vld2.32	{d0, d1, d2, d3}, [r4:16], r6
	vld2.32	{d0, d1, d2, d3}, [r4:32], r6
	vld2.32	{d0, d1, d2, d3}, [r4:64], r6
	vld2.32	{d0, d1, d2, d3}, [r4:128], r6
	vld2.32	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vld2.32 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x24,0xf9,0x86,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld2.32 {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vld2.32 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x24,0xf9,0x96,0x03]
@ CHECK: vld2.32 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x24,0xf9,0xa6,0x03]
@ CHECK: vld2.32 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x24,0xf9,0xb6,0x03]

	vld2.32	{d0[1], d1[1]}, [r4]
	vld2.32	{d0[1], d1[1]}, [r4:16]
	vld2.32	{d0[1], d1[1]}, [r4:32]
	vld2.32	{d0[1], d1[1]}, [r4:64]
	vld2.32	{d0[1], d1[1]}, [r4:128]
	vld2.32	{d0[1], d1[1]}, [r4:256]

@ CHECK: vld2.32 {d0[1], d1[1]}, [r4]    @ encoding: [0xa4,0xf9,0x8f,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d1[1]}, [r4:16]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d1[1]}, [r4:32]
@ CHECK-ERRORS:                                     ^
@ CHECK: vld2.32 {d0[1], d1[1]}, [r4:64] @ encoding: [0xa4,0xf9,0x9f,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d1[1]}, [r4:128]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d1[1]}, [r4:256]
@ CHECK-ERRORS:                                     ^

	vld2.32	{d0[1], d1[1]}, [r4]!
	vld2.32	{d0[1], d1[1]}, [r4:16]!
	vld2.32	{d0[1], d1[1]}, [r4:32]!
	vld2.32	{d0[1], d1[1]}, [r4:64]!
	vld2.32	{d0[1], d1[1]}, [r4:128]!
	vld2.32	{d0[1], d1[1]}, [r4:256]!

@ CHECK: vld2.32 {d0[1], d1[1]}, [r4]!   @ encoding: [0xa4,0xf9,0x8d,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d1[1]}, [r4:16]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d1[1]}, [r4:32]!
@ CHECK-ERRORS:                                     ^
@ CHECK: vld2.32 {d0[1], d1[1]}, [r4:64]! @ encoding: [0xa4,0xf9,0x9d,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d1[1]}, [r4:128]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d1[1]}, [r4:256]!
@ CHECK-ERRORS:                                     ^

	vld2.32	{d0[1], d1[1]}, [r4], r6
	vld2.32	{d0[1], d1[1]}, [r4:16], r6
	vld2.32	{d0[1], d1[1]}, [r4:32], r6
	vld2.32	{d0[1], d1[1]}, [r4:64], r6
	vld2.32	{d0[1], d1[1]}, [r4:128], r6
	vld2.32	{d0[1], d1[1]}, [r4:256], r6

@ CHECK: vld2.32 {d0[1], d1[1]}, [r4], r6 @ encoding: [0xa4,0xf9,0x86,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d1[1]}, [r4:16], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d1[1]}, [r4:32], r6
@ CHECK-ERRORS:                                     ^
@ CHECK: vld2.32 {d0[1], d1[1]}, [r4:64], r6 @ encoding: [0xa4,0xf9,0x96,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d1[1]}, [r4:128], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d1[1]}, [r4:256], r6
@ CHECK-ERRORS:                                     ^

	vld2.32	{d0[1], d2[1]}, [r4]
	vld2.32	{d0[1], d2[1]}, [r4:16]
	vld2.32	{d0[1], d2[1]}, [r4:32]
	vld2.32	{d0[1], d2[1]}, [r4:64]
	vld2.32	{d0[1], d2[1]}, [r4:128]
	vld2.32	{d0[1], d2[1]}, [r4:256]

@ CHECK: vld2.32 {d0[1], d2[1]}, [r4]    @ encoding: [0xa4,0xf9,0xcf,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d2[1]}, [r4:16]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d2[1]}, [r4:32]
@ CHECK-ERRORS:                                     ^
@ CHECK: vld2.32 {d0[1], d2[1]}, [r4:64] @ encoding: [0xa4,0xf9,0xdf,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d2[1]}, [r4:128]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d2[1]}, [r4:256]
@ CHECK-ERRORS:                                     ^

	vld2.32	{d0[1], d2[1]}, [r4]!
	vld2.32	{d0[1], d2[1]}, [r4:16]!
	vld2.32	{d0[1], d2[1]}, [r4:32]!
	vld2.32	{d0[1], d2[1]}, [r4:64]!
	vld2.32	{d0[1], d2[1]}, [r4:128]!
	vld2.32	{d0[1], d2[1]}, [r4:256]!

@ CHECK: vld2.32 {d0[1], d2[1]}, [r4]!   @ encoding: [0xa4,0xf9,0xcd,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d2[1]}, [r4:16]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d2[1]}, [r4:32]!
@ CHECK-ERRORS:                                     ^
@ CHECK: vld2.32 {d0[1], d2[1]}, [r4:64]! @ encoding: [0xa4,0xf9,0xdd,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d2[1]}, [r4:128]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d2[1]}, [r4:256]!
@ CHECK-ERRORS:                                     ^

	vld2.32	{d0[1], d2[1]}, [r4], r6
	vld2.32	{d0[1], d2[1]}, [r4:16], r6
	vld2.32	{d0[1], d2[1]}, [r4:32], r6
	vld2.32	{d0[1], d2[1]}, [r4:64], r6
	vld2.32	{d0[1], d2[1]}, [r4:128], r6
	vld2.32	{d0[1], d2[1]}, [r4:256], r6

@ CHECK: vld2.32 {d0[1], d2[1]}, [r4], r6 @ encoding: [0xa4,0xf9,0xc6,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d2[1]}, [r4:16], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d2[1]}, [r4:32], r6
@ CHECK-ERRORS:                                     ^
@ CHECK: vld2.32 {d0[1], d2[1]}, [r4:64], r6 @ encoding: [0xa4,0xf9,0xd6,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d2[1]}, [r4:128], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[1], d2[1]}, [r4:256], r6
@ CHECK-ERRORS:                                     ^

	vld2.32	{d0[], d1[]}, [r4]
	vld2.32	{d0[], d1[]}, [r4:16]
	vld2.32	{d0[], d1[]}, [r4:32]
	vld2.32	{d0[], d1[]}, [r4:64]
	vld2.32	{d0[], d1[]}, [r4:128]
	vld2.32	{d0[], d1[]}, [r4:256]

@ CHECK: vld2.32 {d0[], d1[]}, [r4]      @ encoding: [0xa4,0xf9,0x8f,0x0d]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d1[]}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d1[]}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld2.32 {d0[], d1[]}, [r4:64]   @ encoding: [0xa4,0xf9,0x9f,0x0d]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d1[]}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d1[]}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld2.32	{d0[], d1[]}, [r4]!
	vld2.32	{d0[], d1[]}, [r4:16]!
	vld2.32	{d0[], d1[]}, [r4:32]!
	vld2.32	{d0[], d1[]}, [r4:64]!
	vld2.32	{d0[], d1[]}, [r4:128]!
	vld2.32	{d0[], d1[]}, [r4:256]!

@ CHECK: vld2.32 {d0[], d1[]}, [r4]!     @ encoding: [0xa4,0xf9,0x8d,0x0d]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d1[]}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d1[]}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld2.32 {d0[], d1[]}, [r4:64]!  @ encoding: [0xa4,0xf9,0x9d,0x0d]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d1[]}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d1[]}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld2.32	{d0[], d1[]}, [r4], r6
	vld2.32	{d0[], d1[]}, [r4:16], r6
	vld2.32	{d0[], d1[]}, [r4:32], r6
	vld2.32	{d0[], d1[]}, [r4:64], r6
	vld2.32	{d0[], d1[]}, [r4:128], r6
	vld2.32	{d0[], d1[]}, [r4:256], r6

@ CHECK: vld2.32 {d0[], d1[]}, [r4], r6  @ encoding: [0xa4,0xf9,0x86,0x0d]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d1[]}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d1[]}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld2.32 {d0[], d1[]}, [r4:64], r6 @ encoding: [0xa4,0xf9,0x96,0x0d]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d1[]}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d1[]}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld2.32	{d0[], d2[]}, [r4]
	vld2.32	{d0[], d2[]}, [r4:16]
	vld2.32	{d0[], d2[]}, [r4:32]
	vld2.32	{d0[], d2[]}, [r4:64]
	vld2.32	{d0[], d2[]}, [r4:128]
	vld2.32	{d0[], d2[]}, [r4:256]

@ CHECK: vld2.32 {d0[], d2[]}, [r4]      @ encoding: [0xa4,0xf9,0xaf,0x0d]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d2[]}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d2[]}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld2.32 {d0[], d2[]}, [r4:64]   @ encoding: [0xa4,0xf9,0xbf,0x0d]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d2[]}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d2[]}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld2.32	{d0[], d2[]}, [r4]!
	vld2.32	{d0[], d2[]}, [r4:16]!
	vld2.32	{d0[], d2[]}, [r4:32]!
	vld2.32	{d0[], d2[]}, [r4:64]!
	vld2.32	{d0[], d2[]}, [r4:128]!
	vld2.32	{d0[], d2[]}, [r4:256]!

@ CHECK: vld2.32 {d0[], d2[]}, [r4]!     @ encoding: [0xa4,0xf9,0xad,0x0d]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d2[]}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d2[]}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld2.32 {d0[], d2[]}, [r4:64]!  @ encoding: [0xa4,0xf9,0xbd,0x0d]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d2[]}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d2[]}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld2.32	{d0[], d2[]}, [r4], r6
	vld2.32	{d0[], d2[]}, [r4:16], r6
	vld2.32	{d0[], d2[]}, [r4:32], r6
	vld2.32	{d0[], d2[]}, [r4:64], r6
	vld2.32	{d0[], d2[]}, [r4:128], r6
	vld2.32	{d0[], d2[]}, [r4:256], r6

@ CHECK: vld2.32 {d0[], d2[]}, [r4], r6  @ encoding: [0xa4,0xf9,0xa6,0x0d]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d2[]}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d2[]}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld2.32 {d0[], d2[]}, [r4:64], r6 @ encoding: [0xa4,0xf9,0xb6,0x0d]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d2[]}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld2.32 {d0[], d2[]}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld3.8	{d0, d1, d2}, [r4]
	vld3.8	{d0, d1, d2}, [r4:16]
	vld3.8	{d0, d1, d2}, [r4:32]
	vld3.8	{d0, d1, d2}, [r4:64]
	vld3.8	{d0, d1, d2}, [r4:128]
	vld3.8	{d0, d1, d2}, [r4:256]

@ CHECK: vld3.8 {d0, d1, d2}, [r4]      @ encoding: [0x24,0xf9,0x0f,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.8 {d0, d1, d2}, [r4:64]   @ encoding: [0x24,0xf9,0x1f,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld3.8	{d0, d1, d2}, [r4]!
	vld3.8	{d0, d1, d2}, [r4:16]!
	vld3.8	{d0, d1, d2}, [r4:32]!
	vld3.8	{d0, d1, d2}, [r4:64]!
	vld3.8	{d0, d1, d2}, [r4:128]!
	vld3.8	{d0, d1, d2}, [r4:256]!

@ CHECK: vld3.8 {d0, d1, d2}, [r4]!     @ encoding: [0x24,0xf9,0x0d,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.8 {d0, d1, d2}, [r4:64]!  @ encoding: [0x24,0xf9,0x1d,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld3.8	{d0, d1, d2}, [r4], r6
	vld3.8	{d0, d1, d2}, [r4:16], r6
	vld3.8	{d0, d1, d2}, [r4:32], r6
	vld3.8	{d0, d1, d2}, [r4:64], r6
	vld3.8	{d0, d1, d2}, [r4:128], r6
	vld3.8	{d0, d1, d2}, [r4:256], r6

@ CHECK: vld3.8 {d0, d1, d2}, [r4], r6  @ encoding: [0x24,0xf9,0x06,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.8 {d0, d1, d2}, [r4:64], r6 @ encoding: [0x24,0xf9,0x16,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld3.8	{d0, d2, d4}, [r4]
	vld3.8	{d0, d2, d4}, [r4:16]
	vld3.8	{d0, d2, d4}, [r4:32]
	vld3.8	{d0, d2, d4}, [r4:64]
	vld3.8	{d0, d2, d4}, [r4:128]
	vld3.8	{d0, d2, d4}, [r4:256]

@ CHECK: vld3.8 {d0, d2, d4}, [r4]      @ encoding: [0x24,0xf9,0x0f,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d2, d4}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d2, d4}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.8 {d0, d2, d4}, [r4:64]   @ encoding: [0x24,0xf9,0x1f,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d2, d4}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d2, d4}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld3.8	{d0, d2, d4}, [r4]!
	vld3.8	{d0, d2, d4}, [r4:16]!
	vld3.8	{d0, d2, d4}, [r4:32]!
	vld3.8	{d0, d2, d4}, [r4:64]!
	vld3.8	{d0, d2, d4}, [r4:128]!
	vld3.8	{d0, d2, d4}, [r4:256]!

@ CHECK: vld3.8 {d0, d2, d4}, [r4]!     @ encoding: [0x24,0xf9,0x0d,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d2, d4}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d2, d4}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.8 {d0, d2, d4}, [r4:64]!  @ encoding: [0x24,0xf9,0x1d,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d2, d4}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d2, d4}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld3.8	{d0, d2, d4}, [r4], r6
	vld3.8	{d0, d2, d4}, [r4:16], r6
	vld3.8	{d0, d2, d4}, [r4:32], r6
	vld3.8	{d0, d2, d4}, [r4:64], r6
	vld3.8	{d0, d2, d4}, [r4:128], r6
	vld3.8	{d0, d2, d4}, [r4:256], r6

@ CHECK: vld3.8 {d0, d2, d4}, [r4], r6  @ encoding: [0x24,0xf9,0x06,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d2, d4}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d2, d4}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.8 {d0, d2, d4}, [r4:64], r6 @ encoding: [0x24,0xf9,0x16,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d2, d4}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.8  {d0, d2, d4}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld3.8	{d0[1], d1[1], d2[1]}, [r4]
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:16]
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:32]
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:64]
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:128]
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:256]

@ CHECK: vld3.8 {d0[1], d1[1], d2[1]}, [r4] @ encoding: [0xa4,0xf9,0x2f,0x02]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:16]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:32]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:64]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:128]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:256]
@ CHECK-ERRORS:                                            ^

	vld3.8	{d0[1], d1[1], d2[1]}, [r4]!
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:16]!
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:32]!
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:64]!
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:128]!
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:256]!

@ CHECK: vld3.8 {d0[1], d1[1], d2[1]}, [r4]! @ encoding: [0xa4,0xf9,0x2d,0x02]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:16]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:32]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:64]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:128]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:256]!
@ CHECK-ERRORS:                                            ^

	vld3.8	{d0[1], d1[1], d2[1]}, [r4], r6
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:16], r6
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:32], r6
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:64], r6
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:128], r6
	vld3.8	{d0[1], d1[1], d2[1]}, [r4:256], r6

@ CHECK: vld3.8 {d0[1], d1[1], d2[1]}, [r4], r6 @ encoding: [0xa4,0xf9,0x26,0x02]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:16], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:32], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:64], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:128], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[1], d1[1], d2[1]}, [r4:256], r6
@ CHECK-ERRORS:                                            ^

	vld3.8	{d0[], d1[], d2[]}, [r4]
	vld3.8	{d0[], d1[], d2[]}, [r4:16]
	vld3.8	{d0[], d1[], d2[]}, [r4:32]
	vld3.8	{d0[], d1[], d2[]}, [r4:64]
	vld3.8	{d0[], d1[], d2[]}, [r4:128]
	vld3.8	{d0[], d1[], d2[]}, [r4:256]

@ CHECK: vld3.8 {d0[], d1[], d2[]}, [r4] @ encoding: [0xa4,0xf9,0x0f,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:16]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:32]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:64]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:128]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:256]
@ CHECK-ERRORS:                                         ^

	vld3.8	{d0[], d1[], d2[]}, [r4]!
	vld3.8	{d0[], d1[], d2[]}, [r4:16]!
	vld3.8	{d0[], d1[], d2[]}, [r4:32]!
	vld3.8	{d0[], d1[], d2[]}, [r4:64]!
	vld3.8	{d0[], d1[], d2[]}, [r4:128]!
	vld3.8	{d0[], d1[], d2[]}, [r4:256]!

@ CHECK: vld3.8 {d0[], d1[], d2[]}, [r4]! @ encoding: [0xa4,0xf9,0x0d,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:16]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:32]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:64]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:128]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:256]!
@ CHECK-ERRORS:                                         ^

	vld3.8	{d0[], d1[], d2[]}, [r4], r6
	vld3.8	{d0[], d1[], d2[]}, [r4:16], r6
	vld3.8	{d0[], d1[], d2[]}, [r4:32], r6
	vld3.8	{d0[], d1[], d2[]}, [r4:64], r6
	vld3.8	{d0[], d1[], d2[]}, [r4:128], r6
	vld3.8	{d0[], d1[], d2[]}, [r4:256], r6

@ CHECK: vld3.8 {d0[], d1[], d2[]}, [r4], r6 @ encoding: [0xa4,0xf9,0x06,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:16], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:32], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:64], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:128], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d1[], d2[]}, [r4:256], r6
@ CHECK-ERRORS:                                         ^

	vld3.8	{d0[], d2[], d4[]}, [r4]
	vld3.8	{d0[], d2[], d4[]}, [r4:16]
	vld3.8	{d0[], d2[], d4[]}, [r4:32]
	vld3.8	{d0[], d2[], d4[]}, [r4:64]
	vld3.8	{d0[], d2[], d4[]}, [r4:128]
	vld3.8	{d0[], d2[], d4[]}, [r4:256]

@ CHECK: vld3.8 {d0[], d2[], d4[]}, [r4] @ encoding: [0xa4,0xf9,0x2f,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:16]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:32]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:64]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:128]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:256]
@ CHECK-ERRORS:                                         ^

	vld3.8	{d0[], d2[], d4[]}, [r4]!
	vld3.8	{d0[], d2[], d4[]}, [r4:16]!
	vld3.8	{d0[], d2[], d4[]}, [r4:32]!
	vld3.8	{d0[], d2[], d4[]}, [r4:64]!
	vld3.8	{d0[], d2[], d4[]}, [r4:128]!
	vld3.8	{d0[], d2[], d4[]}, [r4:256]!

@ CHECK: vld3.8 {d0[], d1[], d2[]}, [r4]! @ encoding: [0xa4,0xf9,0x2d,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:16]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:32]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:64]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:128]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:256]!
@ CHECK-ERRORS:                                         ^

	vld3.8	{d0[], d2[], d4[]}, [r4], r6
	vld3.8	{d0[], d2[], d4[]}, [r4:16], r6
	vld3.8	{d0[], d2[], d4[]}, [r4:32], r6
	vld3.8	{d0[], d2[], d4[]}, [r4:64], r6
	vld3.8	{d0[], d2[], d4[]}, [r4:128], r6
	vld3.8	{d0[], d2[], d4[]}, [r4:256], r6

@ CHECK: vld3.8 {d0[], d2[], d4[]}, [r4], r6 @ encoding: [0xa4,0xf9,0x26,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:16], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:32], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:64], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:128], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.8  {d0[], d2[], d4[]}, [r4:256], r6
@ CHECK-ERRORS:                                         ^

	vld3.16	{d0, d1, d2}, [r4]
	vld3.16	{d0, d1, d2}, [r4:16]
	vld3.16	{d0, d1, d2}, [r4:32]
	vld3.16	{d0, d1, d2}, [r4:64]
	vld3.16	{d0, d1, d2}, [r4:128]
	vld3.16	{d0, d1, d2}, [r4:256]

@ CHECK: vld3.16 {d0, d1, d2}, [r4]      @ encoding: [0x24,0xf9,0x4f,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.16 {d0, d1, d2}, [r4:64]   @ encoding: [0x24,0xf9,0x5f,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld3.16	{d0, d1, d2}, [r4]!
	vld3.16	{d0, d1, d2}, [r4:16]!
	vld3.16	{d0, d1, d2}, [r4:32]!
	vld3.16	{d0, d1, d2}, [r4:64]!
	vld3.16	{d0, d1, d2}, [r4:128]!
	vld3.16	{d0, d1, d2}, [r4:256]!

@ CHECK: vld3.16 {d0, d1, d2}, [r4]!     @ encoding: [0x24,0xf9,0x4d,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.16 {d0, d1, d2}, [r4:64]!  @ encoding: [0x24,0xf9,0x5d,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld3.16	{d0, d1, d2}, [r4], r6
	vld3.16	{d0, d1, d2}, [r4:16], r6
	vld3.16	{d0, d1, d2}, [r4:32], r6
	vld3.16	{d0, d1, d2}, [r4:64], r6
	vld3.16	{d0, d1, d2}, [r4:128], r6
	vld3.16	{d0, d1, d2}, [r4:256], r6

@ CHECK: vld3.16 {d0, d1, d2}, [r4], r6  @ encoding: [0x24,0xf9,0x46,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.16 {d0, d1, d2}, [r4:64], r6 @ encoding: [0x24,0xf9,0x56,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld3.16	{d0, d2, d4}, [r4]
	vld3.16	{d0, d2, d4}, [r4:16]
	vld3.16	{d0, d2, d4}, [r4:32]
	vld3.16	{d0, d2, d4}, [r4:64]
	vld3.16	{d0, d2, d4}, [r4:128]
	vld3.16	{d0, d2, d4}, [r4:256]

@ CHECK: vld3.16 {d0, d2, d4}, [r4]      @ encoding: [0x24,0xf9,0x4f,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d2, d4}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d2, d4}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.16 {d0, d2, d4}, [r4:64]   @ encoding: [0x24,0xf9,0x5f,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d2, d4}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d2, d4}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld3.16	{d0, d2, d4}, [r4]!
	vld3.16	{d0, d2, d4}, [r4:16]!
	vld3.16	{d0, d2, d4}, [r4:32]!
	vld3.16	{d0, d2, d4}, [r4:64]!
	vld3.16	{d0, d2, d4}, [r4:128]!
	vld3.16	{d0, d2, d4}, [r4:256]!

@ CHECK: vld3.16 {d0, d2, d4}, [r4]!     @ encoding: [0x24,0xf9,0x4d,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d2, d4}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d2, d4}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.16 {d0, d2, d4}, [r4:64]!  @ encoding: [0x24,0xf9,0x5d,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d2, d4}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d2, d4}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld3.16	{d0, d2, d4}, [r4], r6
	vld3.16	{d0, d2, d4}, [r4:16], r6
	vld3.16	{d0, d2, d4}, [r4:32], r6
	vld3.16	{d0, d2, d4}, [r4:64], r6
	vld3.16	{d0, d2, d4}, [r4:128], r6
	vld3.16	{d0, d2, d4}, [r4:256], r6

@ CHECK: vld3.16 {d0, d2, d4}, [r4], r6  @ encoding: [0x24,0xf9,0x46,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d2, d4}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d2, d4}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.16 {d0, d2, d4}, [r4:64], r6 @ encoding: [0x24,0xf9,0x56,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d2, d4}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.16 {d0, d2, d4}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld3.16	{d0[1], d1[1], d2[1]}, [r4]
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:16]
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:32]
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:64]
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:128]
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:256]

@ CHECK: vld3.16 {d0[1], d1[1], d2[1]}, [r4] @ encoding: [0xa4,0xf9,0x4f,0x06]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:16]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:32]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:64]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:128]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:256]
@ CHECK-ERRORS:                                            ^

	vld3.16	{d0[1], d1[1], d2[1]}, [r4]!
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:16]!
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:32]!
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:64]!
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:128]!
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:256]!

@ CHECK: vld3.16 {d0[1], d1[1], d2[1]}, [r4]! @ encoding: [0xa4,0xf9,0x4d,0x06]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:16]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:32]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:64]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:128]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:256]!
@ CHECK-ERRORS:                                            ^

	vld3.16	{d0[1], d1[1], d2[1]}, [r4], r6
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:16], r6
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:32], r6
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:64], r6
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:128], r6
	vld3.16	{d0[1], d1[1], d2[1]}, [r4:256], r6

@ CHECK: vld3.16 {d0[1], d1[1], d2[1]}, [r4], r6 @ encoding: [0xa4,0xf9,0x46,0x06]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:16], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:32], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:64], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:128], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d1[1], d2[1]}, [r4:256], r6
@ CHECK-ERRORS:                                            ^

	vld3.16	{d0[1], d2[1], d4[1]}, [r4]
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:16]
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:32]
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:64]
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:128]
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:256]

@ CHECK: vld3.16 {d0[1], d2[1], d4[1]}, [r4] @ encoding: [0xa4,0xf9,0x6f,0x06]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:16]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:32]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:64]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:128]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:256]
@ CHECK-ERRORS:                                            ^

	vld3.16	{d0[1], d2[1], d4[1]}, [r4]!
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:16]!
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:32]!
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:64]!
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:128]!
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:256]!

@ CHECK: vld3.16 {d0[1], d1[1], d2[1]}, [r4]! @ encoding: [0xa4,0xf9,0x6d,0x06]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:16]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:32]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:64]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:128]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:256]!
@ CHECK-ERRORS:                                            ^

	vld3.16	{d0[1], d2[1], d4[1]}, [r4], r6
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:16], r6
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:32], r6
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:64], r6
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:128], r6
	vld3.16	{d0[1], d2[1], d4[1]}, [r4:256], r6

@ CHECK: vld3.16 {d0[1], d2[1], d4[1]}, [r4], r6 @ encoding: [0xa4,0xf9,0x66,0x06]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:16], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:32], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:64], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:128], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[1], d2[1], d4[1]}, [r4:256], r6
@ CHECK-ERRORS:                                            ^

	vld3.16	{d0[], d1[], d2[]}, [r4]
	vld3.16	{d0[], d1[], d2[]}, [r4:16]
	vld3.16	{d0[], d1[], d2[]}, [r4:32]
	vld3.16	{d0[], d1[], d2[]}, [r4:64]
	vld3.16	{d0[], d1[], d2[]}, [r4:128]
	vld3.16	{d0[], d1[], d2[]}, [r4:256]

@ CHECK: vld3.16 {d0[], d1[], d2[]}, [r4] @ encoding: [0xa4,0xf9,0x4f,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:16]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:32]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:64]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:128]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:256]
@ CHECK-ERRORS:                                         ^

	vld3.16	{d0[], d1[], d2[]}, [r4]!
	vld3.16	{d0[], d1[], d2[]}, [r4:16]!
	vld3.16	{d0[], d1[], d2[]}, [r4:32]!
	vld3.16	{d0[], d1[], d2[]}, [r4:64]!
	vld3.16	{d0[], d1[], d2[]}, [r4:128]!
	vld3.16	{d0[], d1[], d2[]}, [r4:256]!

@ CHECK: vld3.16 {d0[], d1[], d2[]}, [r4]! @ encoding: [0xa4,0xf9,0x4d,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:16]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:32]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:64]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:128]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:256]!
@ CHECK-ERRORS:                                         ^

	vld3.16	{d0[], d1[], d2[]}, [r4], r6
	vld3.16	{d0[], d1[], d2[]}, [r4:16], r6
	vld3.16	{d0[], d1[], d2[]}, [r4:32], r6
	vld3.16	{d0[], d1[], d2[]}, [r4:64], r6
	vld3.16	{d0[], d1[], d2[]}, [r4:128], r6
	vld3.16	{d0[], d1[], d2[]}, [r4:256], r6

@ CHECK: vld3.16 {d0[], d1[], d2[]}, [r4], r6 @ encoding: [0xa4,0xf9,0x46,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:16], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:32], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:64], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:128], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d1[], d2[]}, [r4:256], r6
@ CHECK-ERRORS:                                         ^

	vld3.16	{d0[], d2[], d4[]}, [r4]
	vld3.16	{d0[], d2[], d4[]}, [r4:16]
	vld3.16	{d0[], d2[], d4[]}, [r4:32]
	vld3.16	{d0[], d2[], d4[]}, [r4:64]
	vld3.16	{d0[], d2[], d4[]}, [r4:128]
	vld3.16	{d0[], d2[], d4[]}, [r4:256]

@ CHECK: vld3.16 {d0[], d2[], d4[]}, [r4] @ encoding: [0xa4,0xf9,0x6f,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:16]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:32]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:64]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:128]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:256]
@ CHECK-ERRORS:                                         ^

	vld3.16	{d0[], d2[], d4[]}, [r4]!
	vld3.16	{d0[], d2[], d4[]}, [r4:16]!
	vld3.16	{d0[], d2[], d4[]}, [r4:32]!
	vld3.16	{d0[], d2[], d4[]}, [r4:64]!
	vld3.16	{d0[], d2[], d4[]}, [r4:128]!
	vld3.16	{d0[], d2[], d4[]}, [r4:256]!

@ CHECK: vld3.16 {d0[], d2[], d4[]}, [r4]! @ encoding: [0xa4,0xf9,0x6d,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:16]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:32]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:64]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:128]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:256]!
@ CHECK-ERRORS:                                         ^

	vld3.16	{d0[], d2[], d4[]}, [r4], r6
	vld3.16	{d0[], d2[], d4[]}, [r4:16], r6
	vld3.16	{d0[], d2[], d4[]}, [r4:32], r6
	vld3.16	{d0[], d2[], d4[]}, [r4:64], r6
	vld3.16	{d0[], d2[], d4[]}, [r4:128], r6
	vld3.16	{d0[], d2[], d4[]}, [r4:256], r6

@ CHECK: vld3.16 {d0[], d2[], d4[]}, [r4], r6 @ encoding: [0xa4,0xf9,0x66,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:16], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:32], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:64], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:128], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.16 {d0[], d2[], d4[]}, [r4:256], r6

	vld3.32	{d0, d1, d2}, [r4]
	vld3.32	{d0, d1, d2}, [r4:16]
	vld3.32	{d0, d1, d2}, [r4:32]
	vld3.32	{d0, d1, d2}, [r4:64]
	vld3.32	{d0, d1, d2}, [r4:128]
	vld3.32	{d0, d1, d2}, [r4:256]

@ CHECK: vld3.32 {d0, d1, d2}, [r4]      @ encoding: [0x24,0xf9,0x8f,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.32 {d0, d1, d2}, [r4:64]   @ encoding: [0x24,0xf9,0x9f,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld3.32	{d0, d1, d2}, [r4]!
	vld3.32	{d0, d1, d2}, [r4:16]!
	vld3.32	{d0, d1, d2}, [r4:32]!
	vld3.32	{d0, d1, d2}, [r4:64]!
	vld3.32	{d0, d1, d2}, [r4:128]!
	vld3.32	{d0, d1, d2}, [r4:256]!

@ CHECK: vld3.32 {d0, d1, d2}, [r4]!     @ encoding: [0x24,0xf9,0x8d,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.32 {d0, d1, d2}, [r4:64]!  @ encoding: [0x24,0xf9,0x9d,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld3.32	{d0, d1, d2}, [r4], r6
	vld3.32	{d0, d1, d2}, [r4:16], r6
	vld3.32	{d0, d1, d2}, [r4:32], r6
	vld3.32	{d0, d1, d2}, [r4:64], r6
	vld3.32	{d0, d1, d2}, [r4:128], r6
	vld3.32	{d0, d1, d2}, [r4:256], r6

@ CHECK: vld3.32 {d0, d1, d2}, [r4], r6  @ encoding: [0x24,0xf9,0x86,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.32 {d0, d1, d2}, [r4:64], r6 @ encoding: [0x24,0xf9,0x96,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld3.32	{d0, d2, d4}, [r4]
	vld3.32	{d0, d2, d4}, [r4:16]
	vld3.32	{d0, d2, d4}, [r4:32]
	vld3.32	{d0, d2, d4}, [r4:64]
	vld3.32	{d0, d2, d4}, [r4:128]
	vld3.32	{d0, d2, d4}, [r4:256]

@ CHECK: vld3.32 {d0, d2, d4}, [r4]      @ encoding: [0x24,0xf9,0x8f,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d2, d4}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d2, d4}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.32 {d0, d2, d4}, [r4:64]   @ encoding: [0x24,0xf9,0x9f,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d2, d4}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d2, d4}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vld3.32	{d0, d2, d4}, [r4]!
	vld3.32	{d0, d2, d4}, [r4:16]!
	vld3.32	{d0, d2, d4}, [r4:32]!
	vld3.32	{d0, d2, d4}, [r4:64]!
	vld3.32	{d0, d2, d4}, [r4:128]!
	vld3.32	{d0, d2, d4}, [r4:256]!

@ CHECK: vld3.32 {d0, d2, d4}, [r4]!     @ encoding: [0x24,0xf9,0x8d,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d2, d4}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d2, d4}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.32 {d0, d2, d4}, [r4:64]!  @ encoding: [0x24,0xf9,0x9d,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d2, d4}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d2, d4}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vld3.32	{d0, d2, d4}, [r4], r6
	vld3.32	{d0, d2, d4}, [r4:16], r6
	vld3.32	{d0, d2, d4}, [r4:32], r6
	vld3.32	{d0, d2, d4}, [r4:64], r6
	vld3.32	{d0, d2, d4}, [r4:128], r6
	vld3.32	{d0, d2, d4}, [r4:256], r6

@ CHECK: vld3.32 {d0, d2, d4}, [r4], r6  @ encoding: [0x24,0xf9,0x86,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d2, d4}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d2, d4}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vld3.32 {d0, d2, d4}, [r4:64], r6 @ encoding: [0x24,0xf9,0x96,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d2, d4}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld3.32 {d0, d2, d4}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vld3.32	{d0[1], d1[1], d2[1]}, [r4]
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:16]
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:32]
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:64]
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:128]
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:256]

@ CHECK: vld3.32 {d0[1], d1[1], d2[1]}, [r4] @ encoding: [0xa4,0xf9,0x8f,0x0a]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:16]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:32]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:64]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:128]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:256]
@ CHECK-ERRORS:                                            ^

	vld3.32	{d0[1], d1[1], d2[1]}, [r4]!
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:16]!
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:32]!
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:64]!
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:128]!
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:256]!

@ CHECK: vld3.32 {d0[1], d1[1], d2[1]}, [r4]! @ encoding: [0xa4,0xf9,0x8d,0x0a]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:16]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:32]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:64]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:128]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:256]!
@ CHECK-ERRORS:                                            ^

	vld3.32	{d0[1], d1[1], d2[1]}, [r4], r6
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:16], r6
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:32], r6
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:64], r6
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:128], r6
	vld3.32	{d0[1], d1[1], d2[1]}, [r4:256], r6

@ CHECK: vld3.32 {d0[1], d1[1], d2[1]}, [r4], r6 @ encoding: [0xa4,0xf9,0x86,0x0a]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:16], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:32], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:64], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:128], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d1[1], d2[1]}, [r4:256], r6
@ CHECK-ERRORS:                                            ^

	vld3.32	{d0[1], d2[1], d4[1]}, [r4]
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:16]
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:32]
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:64]
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:128]
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:256]

@ CHECK: vld3.32 {d0[1], d2[1], d4[1]}, [r4] @ encoding: [0xa4,0xf9,0xcf,0x0a]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:16]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:32]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:64]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:128]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:256]
@ CHECK-ERRORS:                                            ^

	vld3.32	{d0[1], d2[1], d4[1]}, [r4]!
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:16]!
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:32]!
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:64]!
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:128]!
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:256]!

@ CHECK: vld3.32 {d0[1], d2[1], d4[1]}, [r4]! @ encoding: [0xa4,0xf9,0xcd,0x0a]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:16]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:32]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:64]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:128]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:256]!
@ CHECK-ERRORS:                                            ^

	vld3.32	{d0[1], d2[1], d4[1]}, [r4], r6
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:16], r6
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:32], r6
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:64], r6
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:128], r6
	vld3.32	{d0[1], d2[1], d4[1]}, [r4:256], r6

@ CHECK: vld3.32 {d0[1], d2[1], d4[1]}, [r4], r6 @ encoding: [0xa4,0xf9,0xc6,0x0a]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:16], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:32], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:64], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:128], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[1], d2[1], d4[1]}, [r4:256], r6
@ CHECK-ERRORS:                                            ^

	vld3.32	{d0[], d1[], d2[]}, [r4]
	vld3.32	{d0[], d1[], d2[]}, [r4:16]
	vld3.32	{d0[], d1[], d2[]}, [r4:32]
	vld3.32	{d0[], d1[], d2[]}, [r4:64]
	vld3.32	{d0[], d1[], d2[]}, [r4:128]
	vld3.32	{d0[], d1[], d2[]}, [r4:256]

@ CHECK: vld3.32 {d0[], d1[], d2[]}, [r4] @ encoding: [0xa4,0xf9,0x8f,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:16]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:32]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:64]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:128]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:256]
@ CHECK-ERRORS:                                         ^

	vld3.32	{d0[], d1[], d2[]}, [r4]!
	vld3.32	{d0[], d1[], d2[]}, [r4:16]!
	vld3.32	{d0[], d1[], d2[]}, [r4:32]!
	vld3.32	{d0[], d1[], d2[]}, [r4:64]!
	vld3.32	{d0[], d1[], d2[]}, [r4:128]!
	vld3.32	{d0[], d1[], d2[]}, [r4:256]!

@ CHECK: vld3.32 {d0[], d1[], d2[]}, [r4]! @ encoding: [0xa4,0xf9,0x8d,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:16]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:32]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:64]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:128]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:256]!
@ CHECK-ERRORS:                                         ^

	vld3.32	{d0[], d1[], d2[]}, [r4], r6
	vld3.32	{d0[], d1[], d2[]}, [r4:16], r6
	vld3.32	{d0[], d1[], d2[]}, [r4:32], r6
	vld3.32	{d0[], d1[], d2[]}, [r4:64], r6
	vld3.32	{d0[], d1[], d2[]}, [r4:128], r6
	vld3.32	{d0[], d1[], d2[]}, [r4:256], r6

@ CHECK: vld3.32 {d0[], d1[], d2[]}, [r4], r6 @ encoding: [0xa4,0xf9,0x86,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:16], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:32], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:64], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:128], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d1[], d2[]}, [r4:256], r6
@ CHECK-ERRORS:                                         ^

	vld3.32	{d0[], d2[], d4[]}, [r4]
	vld3.32	{d0[], d2[], d4[]}, [r4:16]
	vld3.32	{d0[], d2[], d4[]}, [r4:32]
	vld3.32	{d0[], d2[], d4[]}, [r4:64]
	vld3.32	{d0[], d2[], d4[]}, [r4:128]
	vld3.32	{d0[], d2[], d4[]}, [r4:256]

@ CHECK: vld3.32 {d0[], d2[], d4[]}, [r4] @ encoding: [0xa4,0xf9,0xaf,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:16]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:32]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:64]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:128]
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:256]
@ CHECK-ERRORS:                                         ^

	vld3.32	{d0[], d2[], d4[]}, [r4]!
	vld3.32	{d0[], d2[], d4[]}, [r4:16]!
	vld3.32	{d0[], d2[], d4[]}, [r4:32]!
	vld3.32	{d0[], d2[], d4[]}, [r4:64]!
	vld3.32	{d0[], d2[], d4[]}, [r4:128]!
	vld3.32	{d0[], d2[], d4[]}, [r4:256]!

@ CHECK: vld3.32 {d0[], d2[], d4[]}, [r4]! @ encoding: [0xa4,0xf9,0xad,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:16]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:32]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:64]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:128]!
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:256]!
@ CHECK-ERRORS:                                         ^

	vld3.32	{d0[], d2[], d4[]}, [r4], r6
	vld3.32	{d0[], d2[], d4[]}, [r4:16], r6
	vld3.32	{d0[], d2[], d4[]}, [r4:32], r6
	vld3.32	{d0[], d2[], d4[]}, [r4:64], r6
	vld3.32	{d0[], d2[], d4[]}, [r4:128], r6
	vld3.32	{d0[], d2[], d4[]}, [r4:256], r6

@ CHECK: vld3.32 {d0[], d2[], d4[]}, [r4], r6 @ encoding: [0xa4,0xf9,0xa6,0x0e]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:16], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:32], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:64], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:128], r6
@ CHECK-ERRORS:                                         ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vld3.32 {d0[], d2[], d4[]}, [r4:256], r6
@ CHECK-ERRORS:                                         ^

	vld4.8	{d0, d1, d2, d3}, [r4]
	vld4.8	{d0, d1, d2, d3}, [r4:16]
	vld4.8	{d0, d1, d2, d3}, [r4:32]
	vld4.8	{d0, d1, d2, d3}, [r4:64]
	vld4.8	{d0, d1, d2, d3}, [r4:128]
	vld4.8	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vld4.8 {d0, d1, d2, d3}, [r4]  @ encoding: [0x24,0xf9,0x0f,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.8  {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.8  {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.8 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x24,0xf9,0x1f,0x00]
@ CHECK: vld4.8 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x24,0xf9,0x2f,0x00]
@ CHECK: vld4.8 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x24,0xf9,0x3f,0x00]

	vld4.8	{d0, d1, d2, d3}, [r4]!
	vld4.8	{d0, d1, d2, d3}, [r4:16]!
	vld4.8	{d0, d1, d2, d3}, [r4:32]!
	vld4.8	{d0, d1, d2, d3}, [r4:64]!
	vld4.8	{d0, d1, d2, d3}, [r4:128]!
	vld4.8	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vld4.8 {d0, d1, d2, d3}, [r4]! @ encoding: [0x24,0xf9,0x0d,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.8  {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.8  {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.8 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x24,0xf9,0x1d,0x00]
@ CHECK: vld4.8 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x24,0xf9,0x2d,0x00]
@ CHECK: vld4.8 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x24,0xf9,0x3d,0x00]

	vld4.8	{d0, d1, d2, d3}, [r4], r6
	vld4.8	{d0, d1, d2, d3}, [r4:16], r6
	vld4.8	{d0, d1, d2, d3}, [r4:32], r6
	vld4.8	{d0, d1, d2, d3}, [r4:64], r6
	vld4.8	{d0, d1, d2, d3}, [r4:128], r6
	vld4.8	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vld4.8 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x24,0xf9,0x06,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.8  {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.8  {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.8 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x24,0xf9,0x16,0x00]
@ CHECK: vld4.8 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x24,0xf9,0x26,0x00]
@ CHECK: vld4.8 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x24,0xf9,0x36,0x00]

	vld4.8	{d0, d2, d4, d6}, [r4]
	vld4.8	{d0, d2, d4, d6}, [r4:16]
	vld4.8	{d0, d2, d4, d6}, [r4:32]
	vld4.8	{d0, d2, d4, d6}, [r4:64]
	vld4.8	{d0, d2, d4, d6}, [r4:128]
	vld4.8	{d0, d2, d4, d6}, [r4:256]

@ CHECK: vld4.8 {d0, d2, d4, d6}, [r4]  @ encoding: [0x24,0xf9,0x0f,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.8  {d0, d2, d4, d6}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.8  {d0, d2, d4, d6}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.8 {d0, d2, d4, d6}, [r4:64] @ encoding: [0x24,0xf9,0x1f,0x01]
@ CHECK: vld4.8 {d0, d2, d4, d6}, [r4:128] @ encoding: [0x24,0xf9,0x2f,0x01]
@ CHECK: vld4.8 {d0, d2, d4, d6}, [r4:256] @ encoding: [0x24,0xf9,0x3f,0x01]

	vld4.8	{d0, d2, d4, d6}, [r4]!
	vld4.8	{d0, d2, d4, d6}, [r4:16]!
	vld4.8	{d0, d2, d4, d6}, [r4:32]!
	vld4.8	{d0, d2, d4, d6}, [r4:64]!
	vld4.8	{d0, d2, d4, d6}, [r4:128]!
	vld4.8	{d0, d2, d4, d6}, [r4:256]!

@ CHECK: vld4.8 {d0, d2, d4, d6}, [r4]! @ encoding: [0x24,0xf9,0x0d,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.8  {d0, d2, d4, d6}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.8  {d0, d2, d4, d6}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.8 {d0, d2, d4, d6}, [r4:64]! @ encoding: [0x24,0xf9,0x1d,0x01]
@ CHECK: vld4.8 {d0, d2, d4, d6}, [r4:128]! @ encoding: [0x24,0xf9,0x2d,0x01]
@ CHECK: vld4.8 {d0, d2, d4, d6}, [r4:256]! @ encoding: [0x24,0xf9,0x3d,0x01]

	vld4.8	{d0, d2, d4, d6}, [r4], r6
	vld4.8	{d0, d2, d4, d6}, [r4:16], r6
	vld4.8	{d0, d2, d4, d6}, [r4:32], r6
	vld4.8	{d0, d2, d4, d6}, [r4:64], r6
	vld4.8	{d0, d2, d4, d6}, [r4:128], r6
	vld4.8	{d0, d2, d4, d6}, [r4:256], r6

@ CHECK: vld4.8 {d0, d2, d4, d6}, [r4], r6 @ encoding: [0x24,0xf9,0x06,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.8  {d0, d2, d4, d6}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.8  {d0, d2, d4, d6}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.8 {d0, d2, d4, d6}, [r4:64], r6 @ encoding: [0x24,0xf9,0x16,0x01]
@ CHECK: vld4.8 {d0, d2, d4, d6}, [r4:128], r6 @ encoding: [0x24,0xf9,0x26,0x01]
@ CHECK: vld4.8 {d0, d2, d4, d6}, [r4:256], r6 @ encoding: [0x24,0xf9,0x36,0x01]

	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4]
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:16]
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:32]
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:64]
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:128]
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:256]

@ CHECK: vld4.8 {d0[1], d1[1], d2[1], d3[1]}, [r4] @ encoding: [0xa4,0xf9,0x2f,0x03]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:16]
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.8 {d0[1], d1[1], d2[1], d3[1]}, [r4:32] @ encoding: [0xa4,0xf9,0x3f,0x03]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:64]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:128]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:256]
@ CHECK-ERRORS:                                                   ^

	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4]!
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:32]!
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:64]!
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:128]!
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:256]!

@ CHECK: vld4.8 {d0[1], d1[1], d2[1], d3[1]}, [r4]! @ encoding: [0xa4,0xf9,0x2d,0x03]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.8 {d0[1], d1[1], d2[1], d3[1]}, [r4:32]! @ encoding: [0xa4,0xf9,0x3d,0x03]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:64]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:128]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:256]!
@ CHECK-ERRORS:                                                   ^

	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4], r6
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6
	vld4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6

@ CHECK: vld4.8 {d0[1], d1[1], d2[1], d3[1]}, [r4], r6 @ encoding: [0xa4,0xf9,0x26,0x03]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.8 {d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6 @ encoding: [0xa4,0xf9,0x36,0x03]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6
@ CHECK-ERRORS:                                                   ^

	vld4.8	{d0[], d1[], d2[], d3[]}, [r4]
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:16]
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:32]
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:64]
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:128]
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:256]

@ CHECK: vld4.8 {d0[], d1[], d2[], d3[]}, [r4] @ encoding: [0xa4,0xf9,0x0f,0x0f]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d1[], d2[], d3[]}, [r4:16]
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.8 {d0[], d1[], d2[], d3[]}, [r4:32] @ encoding: [0xa4,0xf9,0x1f,0x0f]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d1[], d2[], d3[]}, [r4:64]
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d1[], d2[], d3[]}, [r4:128]
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d1[], d2[], d3[]}, [r4:256]
@ CHECK-ERRORS:                                               ^

	vld4.8	{d0[], d1[], d2[], d3[]}, [r4]!
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:16]!
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:32]!
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:64]!
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:128]!
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:256]!

@ CHECK: vld4.8 {d0[], d1[], d2[], d3[]}, [r4]! @ encoding: [0xa4,0xf9,0x0d,0x0f]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d1[], d2[], d3[]}, [r4:16]!
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.8 {d0[], d1[], d2[], d3[]}, [r4:32]! @ encoding: [0xa4,0xf9,0x1d,0x0f]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d1[], d2[], d3[]}, [r4:64]!
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d1[], d2[], d3[]}, [r4:128]!
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d1[], d2[], d3[]}, [r4:256]!
@ CHECK-ERRORS:                                               ^

	vld4.8	{d0[], d1[], d2[], d3[]}, [r4], r6
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:16], r6
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:32], r6
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:64], r6
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:128], r6
	vld4.8	{d0[], d1[], d2[], d3[]}, [r4:256], r6

@ CHECK: vld4.8 {d0[], d1[], d2[], d3[]}, [r4], r6 @ encoding: [0xa4,0xf9,0x06,0x0f]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d1[], d2[], d3[]}, [r4:16], r6
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.8 {d0[], d1[], d2[], d3[]}, [r4:32], r6 @ encoding: [0xa4,0xf9,0x16,0x0f]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d1[], d2[], d3[]}, [r4:64], r6
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d1[], d2[], d3[]}, [r4:128], r6
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d1[], d2[], d3[]}, [r4:256], r6
@ CHECK-ERRORS:                                               ^

	vld4.8	{d0[], d2[], d4[], d6[]}, [r4]
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:16]
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:32]
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:64]
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:128]
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:256]

@ CHECK: vld4.8 {d0[], d2[], d4[], d6[]}, [r4] @ encoding: [0xa4,0xf9,0x2f,0x0f]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d2[], d4[], d6[]}, [r4:16]
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.8 {d0[], d2[], d4[], d6[]}, [r4:32] @ encoding: [0xa4,0xf9,0x3f,0x0f]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d2[], d4[], d6[]}, [r4:64]
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d2[], d4[], d6[]}, [r4:128]
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d2[], d4[], d6[]}, [r4:256]
@ CHECK-ERRORS:                                               ^

	vld4.8	{d0[], d2[], d4[], d6[]}, [r4]!
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:16]!
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:32]!
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:64]!
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:128]!
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:256]!

@ CHECK: vld4.8 {d0[], d1[], d2[], d3[]}, [r4]! @ encoding: [0xa4,0xf9,0x2d,0x0f]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d2[], d4[], d6[]}, [r4:16]!
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.8 {d0[], d1[], d2[], d3[]}, [r4:32]! @ encoding: [0xa4,0xf9,0x3d,0x0f]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d2[], d4[], d6[]}, [r4:64]!
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d2[], d4[], d6[]}, [r4:128]!
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d2[], d4[], d6[]}, [r4:256]!
@ CHECK-ERRORS:                                               ^

	vld4.8	{d0[], d2[], d4[], d6[]}, [r4], r6
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:16], r6
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:32], r6
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:64], r6
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:128], r6
	vld4.8	{d0[], d2[], d4[], d6[]}, [r4:256], r6

@ CHECK: vld4.8 {d0[], d2[], d4[], d6[]}, [r4], r6 @ encoding: [0xa4,0xf9,0x26,0x0f]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d2[], d4[], d6[]}, [r4:16], r6
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.8 {d0[], d2[], d4[], d6[]}, [r4:32], r6 @ encoding: [0xa4,0xf9,0x36,0x0f]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d2[], d4[], d6[]}, [r4:64], r6
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d2[], d4[], d6[]}, [r4:128], r6
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vld4.8  {d0[], d2[], d4[], d6[]}, [r4:256], r6
@ CHECK-ERRORS:                                               ^

	vld4.16	{d0, d1, d2, d3}, [r4]
	vld4.16	{d0, d1, d2, d3}, [r4:16]
	vld4.16	{d0, d1, d2, d3}, [r4:32]
	vld4.16	{d0, d1, d2, d3}, [r4:64]
	vld4.16	{d0, d1, d2, d3}, [r4:128]
	vld4.16	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vld4.16 {d0, d1, d2, d3}, [r4]  @ encoding: [0x24,0xf9,0x4f,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.16 {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.16 {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.16 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x24,0xf9,0x5f,0x00]
@ CHECK: vld4.16 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x24,0xf9,0x6f,0x00]
@ CHECK: vld4.16 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x24,0xf9,0x7f,0x00]

	vld4.16	{d0, d1, d2, d3}, [r4]!
	vld4.16	{d0, d1, d2, d3}, [r4:16]!
	vld4.16	{d0, d1, d2, d3}, [r4:32]!
	vld4.16	{d0, d1, d2, d3}, [r4:64]!
	vld4.16	{d0, d1, d2, d3}, [r4:128]!
	vld4.16	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vld4.16 {d0, d1, d2, d3}, [r4]! @ encoding: [0x24,0xf9,0x4d,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.16 {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.16 {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.16 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x24,0xf9,0x5d,0x00]
@ CHECK: vld4.16 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x24,0xf9,0x6d,0x00]
@ CHECK: vld4.16 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x24,0xf9,0x7d,0x00]

	vld4.16	{d0, d1, d2, d3}, [r4], r6
	vld4.16	{d0, d1, d2, d3}, [r4:16], r6
	vld4.16	{d0, d1, d2, d3}, [r4:32], r6
	vld4.16	{d0, d1, d2, d3}, [r4:64], r6
	vld4.16	{d0, d1, d2, d3}, [r4:128], r6
	vld4.16	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vld4.16 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x24,0xf9,0x46,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.16 {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.16 {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.16 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x24,0xf9,0x56,0x00]
@ CHECK: vld4.16 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x24,0xf9,0x66,0x00]
@ CHECK: vld4.16 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x24,0xf9,0x76,0x00]

	vld4.16	{d0, d2, d4, d6}, [r4]
	vld4.16	{d0, d2, d4, d6}, [r4:16]
	vld4.16	{d0, d2, d4, d6}, [r4:32]
	vld4.16	{d0, d2, d4, d6}, [r4:64]
	vld4.16	{d0, d2, d4, d6}, [r4:128]
	vld4.16	{d0, d2, d4, d6}, [r4:256]

@ CHECK: vld4.16 {d0, d2, d4, d6}, [r4]  @ encoding: [0x24,0xf9,0x4f,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.16 {d0, d2, d4, d6}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.16 {d0, d2, d4, d6}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.16 {d0, d2, d4, d6}, [r4:64] @ encoding: [0x24,0xf9,0x5f,0x01]
@ CHECK: vld4.16 {d0, d2, d4, d6}, [r4:128] @ encoding: [0x24,0xf9,0x6f,0x01]
@ CHECK: vld4.16 {d0, d2, d4, d6}, [r4:256] @ encoding: [0x24,0xf9,0x7f,0x01]

	vld4.16	{d0, d2, d4, d6}, [r4]!
	vld4.16	{d0, d2, d4, d6}, [r4:16]!
	vld4.16	{d0, d2, d4, d6}, [r4:32]!
	vld4.16	{d0, d2, d4, d6}, [r4:64]!
	vld4.16	{d0, d2, d4, d6}, [r4:128]!
	vld4.16	{d0, d2, d4, d6}, [r4:256]!

@ CHECK: vld4.16 {d0, d2, d4, d6}, [r4]! @ encoding: [0x24,0xf9,0x4d,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.16 {d0, d2, d4, d6}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.16 {d0, d2, d4, d6}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.16 {d0, d2, d4, d6}, [r4:64]! @ encoding: [0x24,0xf9,0x5d,0x01]
@ CHECK: vld4.16 {d0, d2, d4, d6}, [r4:128]! @ encoding: [0x24,0xf9,0x6d,0x01]
@ CHECK: vld4.16 {d0, d2, d4, d6}, [r4:256]! @ encoding: [0x24,0xf9,0x7d,0x01]

	vld4.16	{d0, d2, d4, d6}, [r4], r6
	vld4.16	{d0, d2, d4, d6}, [r4:16], r6
	vld4.16	{d0, d2, d4, d6}, [r4:32], r6
	vld4.16	{d0, d2, d4, d6}, [r4:64], r6
	vld4.16	{d0, d2, d4, d6}, [r4:128], r6
	vld4.16	{d0, d2, d4, d6}, [r4:256], r6

@ CHECK: vld4.16 {d0, d2, d4, d6}, [r4], r6 @ encoding: [0x24,0xf9,0x46,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.16 {d0, d2, d4, d6}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.16 {d0, d2, d4, d6}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.16 {d0, d2, d4, d6}, [r4:64], r6 @ encoding: [0x24,0xf9,0x56,0x01]
@ CHECK: vld4.16 {d0, d2, d4, d6}, [r4:128], r6 @ encoding: [0x24,0xf9,0x66,0x01]
@ CHECK: vld4.16 {d0, d2, d4, d6}, [r4:256], r6 @ encoding: [0x24,0xf9,0x76,0x01]

	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4]
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:16]
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:32]
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:64]
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:128]
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:256]

@ CHECK: vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4] @ encoding: [0xa4,0xf9,0x4f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:16]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:32]
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:64] @ encoding: [0xa4,0xf9,0x5f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:128]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:256]
@ CHECK-ERRORS:                                                   ^

	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4]!
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:32]!
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:64]!
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:128]!
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:256]!

@ CHECK: vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4]! @ encoding: [0xa4,0xf9,0x4d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:32]!
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:64]! @ encoding: [0xa4,0xf9,0x5d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:128]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:256]!
@ CHECK-ERRORS:                                                   ^

	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4], r6
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6
	vld4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6

@ CHECK: vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4], r6 @ encoding: [0xa4,0xf9,0x46,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6 @ encoding: [0xa4,0xf9,0x56,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6
@ CHECK-ERRORS:                                                   ^

	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4]
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:16]
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:32]
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:64]
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:128]
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:256]

@ CHECK: vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4] @ encoding: [0xa4,0xf9,0x6f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:16]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:32]
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:64] @ encoding: [0xa4,0xf9,0x7f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:128]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:256]
@ CHECK-ERRORS:                                                   ^

	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4]!
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:16]!
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:32]!
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:64]!
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:128]!
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:256]!

@ CHECK: vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4]! @ encoding: [0xa4,0xf9,0x6d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:16]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:32]!
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:64]! @ encoding: [0xa4,0xf9,0x7d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:128]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:256]!
@ CHECK-ERRORS:                                                   ^

	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4], r6
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:16], r6
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:32], r6
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:64], r6
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:128], r6
	vld4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:256], r6

@ CHECK: vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4], r6 @ encoding: [0xa4,0xf9,0x66,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:16], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:32], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:64], r6 @ encoding: [0xa4,0xf9,0x76,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:128], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:256], r6
@ CHECK-ERRORS:                                                   ^

	vld4.16	{d0[], d1[], d2[], d3[]}, [r4]
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:16]
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:32]
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:64]
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:128]
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:256]

@ CHECK: vld4.16 {d0[], d1[], d2[], d3[]}, [r4] @ encoding: [0xa4,0xf9,0x4f,0x0f]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d1[], d2[], d3[]}, [r4:16]
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d1[], d2[], d3[]}, [r4:32]
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.16 {d0[], d1[], d2[], d3[]}, [r4:64] @ encoding: [0xa4,0xf9,0x5f,0x0f]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d1[], d2[], d3[]}, [r4:128]
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d1[], d2[], d3[]}, [r4:256]
@ CHECK-ERRORS:                                               ^

	vld4.16	{d0[], d1[], d2[], d3[]}, [r4]!
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:16]!
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:32]!
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:64]!
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:128]!
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:256]!

@ CHECK: vld4.16 {d0[], d1[], d2[], d3[]}, [r4]! @ encoding: [0xa4,0xf9,0x4d,0x0f]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d1[], d2[], d3[]}, [r4:16]!
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d1[], d2[], d3[]}, [r4:32]!
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.16 {d0[], d1[], d2[], d3[]}, [r4:64]! @ encoding: [0xa4,0xf9,0x5d,0x0f]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d1[], d2[], d3[]}, [r4:128]!
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d1[], d2[], d3[]}, [r4:256]!
@ CHECK-ERRORS:                                               ^

	vld4.16	{d0[], d1[], d2[], d3[]}, [r4], r6
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:16], r6
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:32], r6
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:64], r6
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:128], r6
	vld4.16	{d0[], d1[], d2[], d3[]}, [r4:256], r6

@ CHECK: vld4.16 {d0[], d1[], d2[], d3[]}, [r4], r6 @ encoding: [0xa4,0xf9,0x46,0x0f]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d1[], d2[], d3[]}, [r4:16], r6
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d1[], d2[], d3[]}, [r4:32], r6
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.16 {d0[], d1[], d2[], d3[]}, [r4:64], r6 @ encoding: [0xa4,0xf9,0x56,0x0f]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d1[], d2[], d3[]}, [r4:128], r6
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d1[], d2[], d3[]}, [r4:256], r6
@ CHECK-ERRORS:                                               ^

	vld4.16	{d0[], d2[], d4[], d6[]}, [r4]
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:16]
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:32]
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:64]
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:128]
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:256]

@ CHECK: vld4.16 {d0[], d2[], d4[], d6[]}, [r4] @ encoding: [0xa4,0xf9,0x6f,0x0f]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d2[], d4[], d6[]}, [r4:16]
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d2[], d4[], d6[]}, [r4:32]
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.16 {d0[], d2[], d4[], d6[]}, [r4:64] @ encoding: [0xa4,0xf9,0x7f,0x0f]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d2[], d4[], d6[]}, [r4:128]
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d2[], d4[], d6[]}, [r4:256]
@ CHECK-ERRORS:                                               ^

	vld4.16	{d0[], d2[], d4[], d6[]}, [r4]!
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:16]!
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:32]!
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:64]!
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:128]!
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:256]!

@ CHECK: vld4.16 {d0[], d1[], d2[], d3[]}, [r4]! @ encoding: [0xa4,0xf9,0x6d,0x0f]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d2[], d4[], d6[]}, [r4:16]!
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d2[], d4[], d6[]}, [r4:32]!
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.16 {d0[], d1[], d2[], d3[]}, [r4:64]! @ encoding: [0xa4,0xf9,0x7d,0x0f]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d2[], d4[], d6[]}, [r4:128]!
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d2[], d4[], d6[]}, [r4:256]!
@ CHECK-ERRORS:                                               ^

	vld4.16	{d0[], d2[], d4[], d6[]}, [r4], r6
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:16], r6
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:32], r6
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:64], r6
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:128], r6
	vld4.16	{d0[], d2[], d4[], d6[]}, [r4:256], r6

@ CHECK: vld4.16 {d0[], d2[], d4[], d6[]}, [r4], r6 @ encoding: [0xa4,0xf9,0x66,0x0f]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d2[], d4[], d6[]}, [r4:16], r6
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d2[], d4[], d6[]}, [r4:32], r6
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.16 {d0[], d2[], d4[], d6[]}, [r4:64], r6 @ encoding: [0xa4,0xf9,0x76,0x0f]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d2[], d4[], d6[]}, [r4:128], r6
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vld4.16 {d0[], d2[], d4[], d6[]}, [r4:256], r6
@ CHECK-ERRORS:                                               ^

	vld4.32	{d0, d1, d2, d3}, [r4]
	vld4.32	{d0, d1, d2, d3}, [r4:16]
	vld4.32	{d0, d1, d2, d3}, [r4:32]
	vld4.32	{d0, d1, d2, d3}, [r4:64]
	vld4.32	{d0, d1, d2, d3}, [r4:128]
	vld4.32	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vld4.32 {d0, d1, d2, d3}, [r4]  @ encoding: [0x24,0xf9,0x8f,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.32 {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.32 {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.32 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x24,0xf9,0x9f,0x00]
@ CHECK: vld4.32 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x24,0xf9,0xaf,0x00]
@ CHECK: vld4.32 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x24,0xf9,0xbf,0x00]

	vld4.32	{d0, d1, d2, d3}, [r4]!
	vld4.32	{d0, d1, d2, d3}, [r4:16]!
	vld4.32	{d0, d1, d2, d3}, [r4:32]!
	vld4.32	{d0, d1, d2, d3}, [r4:64]!
	vld4.32	{d0, d1, d2, d3}, [r4:128]!
	vld4.32	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vld4.32 {d0, d1, d2, d3}, [r4]! @ encoding: [0x24,0xf9,0x8d,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.32 {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.32 {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.32 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x24,0xf9,0x9d,0x00]
@ CHECK: vld4.32 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x24,0xf9,0xad,0x00]
@ CHECK: vld4.32 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x24,0xf9,0xbd,0x00]

	vld4.32	{d0, d1, d2, d3}, [r4], r6
	vld4.32	{d0, d1, d2, d3}, [r4:16], r6
	vld4.32	{d0, d1, d2, d3}, [r4:32], r6
	vld4.32	{d0, d1, d2, d3}, [r4:64], r6
	vld4.32	{d0, d1, d2, d3}, [r4:128], r6
	vld4.32	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vld4.32 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x24,0xf9,0x86,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.32 {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.32 {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.32 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x24,0xf9,0x96,0x00]
@ CHECK: vld4.32 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x24,0xf9,0xa6,0x00]
@ CHECK: vld4.32 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x24,0xf9,0xb6,0x00]

	vld4.32	{d0, d2, d4, d6}, [r4]
	vld4.32	{d0, d2, d4, d6}, [r4:16]
	vld4.32	{d0, d2, d4, d6}, [r4:32]
	vld4.32	{d0, d2, d4, d6}, [r4:64]
	vld4.32	{d0, d2, d4, d6}, [r4:128]
	vld4.32	{d0, d2, d4, d6}, [r4:256]

@ CHECK: vld4.32 {d0, d2, d4, d6}, [r4]  @ encoding: [0x24,0xf9,0x8f,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.32 {d0, d2, d4, d6}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.32 {d0, d2, d4, d6}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.32 {d0, d2, d4, d6}, [r4:64] @ encoding: [0x24,0xf9,0x9f,0x01]
@ CHECK: vld4.32 {d0, d2, d4, d6}, [r4:128] @ encoding: [0x24,0xf9,0xaf,0x01]
@ CHECK: vld4.32 {d0, d2, d4, d6}, [r4:256] @ encoding: [0x24,0xf9,0xbf,0x01]

	vld4.32	{d0, d2, d4, d6}, [r4]!
	vld4.32	{d0, d2, d4, d6}, [r4:16]!
	vld4.32	{d0, d2, d4, d6}, [r4:32]!
	vld4.32	{d0, d2, d4, d6}, [r4:64]!
	vld4.32	{d0, d2, d4, d6}, [r4:128]!
	vld4.32	{d0, d2, d4, d6}, [r4:256]!

@ CHECK: vld4.32 {d0, d2, d4, d6}, [r4]! @ encoding: [0x24,0xf9,0x8d,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.32 {d0, d2, d4, d6}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.32 {d0, d2, d4, d6}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.32 {d0, d2, d4, d6}, [r4:64]! @ encoding: [0x24,0xf9,0x9d,0x01]
@ CHECK: vld4.32 {d0, d2, d4, d6}, [r4:128]! @ encoding: [0x24,0xf9,0xad,0x01]
@ CHECK: vld4.32 {d0, d2, d4, d6}, [r4:256]! @ encoding: [0x24,0xf9,0xbd,0x01]

	vld4.32	{d0, d2, d4, d6}, [r4], r6
	vld4.32	{d0, d2, d4, d6}, [r4:16], r6
	vld4.32	{d0, d2, d4, d6}, [r4:32], r6
	vld4.32	{d0, d2, d4, d6}, [r4:64], r6
	vld4.32	{d0, d2, d4, d6}, [r4:128], r6
	vld4.32	{d0, d2, d4, d6}, [r4:256], r6

@ CHECK: vld4.32 {d0, d2, d4, d6}, [r4], r6 @ encoding: [0x24,0xf9,0x86,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.32 {d0, d2, d4, d6}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vld4.32 {d0, d2, d4, d6}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vld4.32 {d0, d2, d4, d6}, [r4:64], r6 @ encoding: [0x24,0xf9,0x96,0x01]
@ CHECK: vld4.32 {d0, d2, d4, d6}, [r4:128], r6 @ encoding: [0x24,0xf9,0xa6,0x01]
@ CHECK: vld4.32 {d0, d2, d4, d6}, [r4:256], r6 @ encoding: [0x24,0xf9,0xb6,0x01]

	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4]
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:16]
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:32]
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:64]
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:128]
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:256]

@ CHECK: vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4] @ encoding: [0xa4,0xf9,0x8f,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:16]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:32]
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:64] @ encoding: [0xa4,0xf9,0x9f,0x0b]
@ CHECK: vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:128] @ encoding: [0xa4,0xf9,0xaf,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:256]
@ CHECK-ERRORS:                                                   ^

	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4]!
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:32]!
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:64]!
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:128]!
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:256]!

@ CHECK: vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4]! @ encoding: [0xa4,0xf9,0x8d,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:32]!
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:64]! @ encoding: [0xa4,0xf9,0x9d,0x0b]
@ CHECK: vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:128]! @ encoding: [0xa4,0xf9,0xad,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:256]!
@ CHECK-ERRORS:                                                   ^

	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4], r6
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6
	vld4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6

@ CHECK: vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4], r6 @ encoding: [0xa4,0xf9,0x86,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6 @ encoding: [0xa4,0xf9,0x96,0x0b]
@ CHECK: vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6 @ encoding: [0xa4,0xf9,0xa6,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6
@ CHECK-ERRORS:                                                   ^

	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4]
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:16]
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:32]
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:64]
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:128]
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:256]

@ CHECK: vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4] @ encoding: [0xa4,0xf9,0xcf,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:16]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:32]
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:64] @ encoding: [0xa4,0xf9,0xdf,0x0b]
@ CHECK: vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:128] @ encoding: [0xa4,0xf9,0xef,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:256]
@ CHECK-ERRORS:                                                   ^

	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4]!
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:16]!
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:32]!
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:64]!
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:128]!
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:256]!

@ CHECK: vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4]! @ encoding: [0xa4,0xf9,0xcd,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:16]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:32]!
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:64]! @ encoding: [0xa4,0xf9,0xdd,0x0b]
@ CHECK: vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:128]! @ encoding: [0xa4,0xf9,0xed,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:256]!
@ CHECK-ERRORS:                                                   ^

	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4], r6
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:16], r6
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:32], r6
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:64], r6
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:128], r6
	vld4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:256], r6

@ CHECK: vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4], r6 @ encoding: [0xa4,0xf9,0xc6,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:16], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:32], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK: vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:64], r6 @ encoding: [0xa4,0xf9,0xd6,0x0b]
@ CHECK: vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:128], r6 @ encoding: [0xa4,0xf9,0xe6,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:256], r6
@ CHECK-ERRORS:                                                   ^

	vld4.32	{d0[], d1[], d2[], d3[]}, [r4]
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:16]
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:32]
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:64]
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:128]
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:256]

@ CHECK: vld4.32 {d0[], d1[], d2[], d3[]}, [r4] @ encoding: [0xa4,0xf9,0x8f,0x0f]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d1[], d2[], d3[]}, [r4:16]
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d1[], d2[], d3[]}, [r4:32]
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.32 {d0[], d1[], d2[], d3[]}, [r4:64] @ encoding: [0xa4,0xf9,0x9f,0x0f]
@ CHECK: vld4.32 {d0[], d1[], d2[], d3[]}, [r4:128] @ encoding: [0xa4,0xf9,0xdf,0x0f]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d1[], d2[], d3[]}, [r4:256]
@ CHECK-ERRORS:                                               ^

	vld4.32	{d0[], d1[], d2[], d3[]}, [r4]!
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:16]!
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:32]!
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:64]!
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:128]!
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:256]!

@ CHECK: vld4.32 {d0[], d1[], d2[], d3[]}, [r4]! @ encoding: [0xa4,0xf9,0x8d,0x0f]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d1[], d2[], d3[]}, [r4:16]!
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d1[], d2[], d3[]}, [r4:32]!
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.32 {d0[], d1[], d2[], d3[]}, [r4:64]! @ encoding: [0xa4,0xf9,0x9d,0x0f]
@ CHECK: vld4.32 {d0[], d1[], d2[], d3[]}, [r4:128]! @ encoding: [0xa4,0xf9,0xdd,0x0f]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d1[], d2[], d3[]}, [r4:256]!
@ CHECK-ERRORS:                                               ^

	vld4.32	{d0[], d1[], d2[], d3[]}, [r4], r6
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:16], r6
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:32], r6
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:64], r6
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:128], r6
	vld4.32	{d0[], d1[], d2[], d3[]}, [r4:256], r6

@ CHECK: vld4.32 {d0[], d1[], d2[], d3[]}, [r4], r6 @ encoding: [0xa4,0xf9,0x86,0x0f]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d1[], d2[], d3[]}, [r4:16], r6
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d1[], d2[], d3[]}, [r4:32], r6
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.32 {d0[], d1[], d2[], d3[]}, [r4:64], r6 @ encoding: [0xa4,0xf9,0x96,0x0f]
@ CHECK: vld4.32 {d0[], d1[], d2[], d3[]}, [r4:128], r6 @ encoding: [0xa4,0xf9,0xd6,0x0f]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d1[], d2[], d3[]}, [r4:256], r6
@ CHECK-ERRORS:                                               ^

	vld4.32	{d0[], d2[], d4[], d6[]}, [r4]
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:16]
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:32]
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:64]
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:128]
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:256]

@ CHECK: vld4.32 {d0[], d2[], d4[], d6[]}, [r4] @ encoding: [0xa4,0xf9,0xaf,0x0f]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d2[], d4[], d6[]}, [r4:16]
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d2[], d4[], d6[]}, [r4:32]
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.32 {d0[], d2[], d4[], d6[]}, [r4:64] @ encoding: [0xa4,0xf9,0xbf,0x0f]
@ CHECK: vld4.32 {d0[], d2[], d4[], d6[]}, [r4:128] @ encoding: [0xa4,0xf9,0xff,0x0f]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d2[], d4[], d6[]}, [r4:256]
@ CHECK-ERRORS:                                               ^

	vld4.32	{d0[], d2[], d4[], d6[]}, [r4]!
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:16]!
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:32]!
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:64]!
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:128]!
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:256]!

@ CHECK: vld4.32 {d0[], d2[], d4[], d6[]}, [r4]! @ encoding: [0xa4,0xf9,0xad,0x0f]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d2[], d4[], d6[]}, [r4:16]!
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d2[], d4[], d6[]}, [r4:32]!
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.32 {d0[], d2[], d4[], d6[]}, [r4:64]! @ encoding: [0xa4,0xf9,0xbd,0x0f]
@ CHECK: vld4.32 {d0[], d2[], d4[], d6[]}, [r4:128]! @ encoding: [0xa4,0xf9,0xfd,0x0f]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d2[], d4[], d6[]}, [r4:256]!
@ CHECK-ERRORS:                                               ^

	vld4.32	{d0[], d2[], d4[], d6[]}, [r4], r6
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:16], r6
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:32], r6
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:64], r6
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:128], r6
	vld4.32	{d0[], d2[], d4[], d6[]}, [r4:256], r6

@ CHECK: vld4.32 {d0[], d2[], d4[], d6[]}, [r4], r6 @ encoding: [0xa4,0xf9,0xa6,0x0f]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d2[], d4[], d6[]}, [r4:16], r6
@ CHECK-ERRORS:                                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d2[], d4[], d6[]}, [r4:32], r6
@ CHECK-ERRORS:                                               ^
@ CHECK: vld4.32 {d0[], d2[], d4[], d6[]}, [r4:64], r6 @ encoding: [0xa4,0xf9,0xb6,0x0f]
@ CHECK: vld4.32 {d0[], d2[], d4[], d6[]}, [r4:128], r6 @ encoding: [0xa4,0xf9,0xf6,0x0f]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vld4.32 {d0[], d2[], d4[], d6[]}, [r4:256], r6
@ CHECK-ERRORS:                                               ^

	vst1.8	{d0}, [r4]
	vst1.8	{d0}, [r4:16]
	vst1.8	{d0}, [r4:32]
	vst1.8	{d0}, [r4:64]
	vst1.8	{d0}, [r4:128]
	vst1.8	{d0}, [r4:256]

@ CHECK: vst1.8 {d0}, [r4]              @ encoding: [0x04,0xf9,0x0f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0}, [r4:16]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0}, [r4:32]
@ CHECK-ERRORS:                           ^
@ CHECK: vst1.8 {d0}, [r4:64]           @ encoding: [0x04,0xf9,0x1f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0}, [r4:128]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0}, [r4:256]
@ CHECK-ERRORS:                           ^

	vst1.8	{d0}, [r4]!
	vst1.8	{d0}, [r4:16]!
	vst1.8	{d0}, [r4:32]!
	vst1.8	{d0}, [r4:64]!
	vst1.8	{d0}, [r4:128]!
	vst1.8	{d0}, [r4:256]!

@ CHECK: vst1.8 {d0}, [r4]!             @ encoding: [0x04,0xf9,0x0d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0}, [r4:16]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0}, [r4:32]!
@ CHECK-ERRORS:                           ^
@ CHECK: vst1.8 {d0}, [r4:64]!          @ encoding: [0x04,0xf9,0x1d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0}, [r4:128]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0}, [r4:256]!
@ CHECK-ERRORS:                           ^

	vst1.8	{d0}, [r4], r6
	vst1.8	{d0}, [r4:16], r6
	vst1.8	{d0}, [r4:32], r6
	vst1.8	{d0}, [r4:64], r6
	vst1.8	{d0}, [r4:128], r6
	vst1.8	{d0}, [r4:256], r6

@ CHECK: vst1.8 {d0}, [r4], r6          @ encoding: [0x04,0xf9,0x06,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0}, [r4:16], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0}, [r4:32], r6
@ CHECK-ERRORS:                           ^
@ CHECK: vst1.8 {d0}, [r4:64], r6       @ encoding: [0x04,0xf9,0x16,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0}, [r4:128], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0}, [r4:256], r6
@ CHECK-ERRORS:                           ^

	vst1.8	{d0, d1}, [r4]
	vst1.8	{d0, d1}, [r4:16]
	vst1.8	{d0, d1}, [r4:32]
	vst1.8	{d0, d1}, [r4:64]
	vst1.8	{d0, d1}, [r4:128]
	vst1.8	{d0, d1}, [r4:256]

@ CHECK: vst1.8 {d0, d1}, [r4]          @ encoding: [0x04,0xf9,0x0f,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vst1.8 {d0, d1}, [r4:64]       @ encoding: [0x04,0xf9,0x1f,0x0a]
@ CHECK: vst1.8 {d0, d1}, [r4:128]      @ encoding: [0x04,0xf9,0x2f,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1}, [r4:256]
@ CHECK-ERRORS:                               ^

	vst1.8	{d0, d1}, [r4]!
	vst1.8	{d0, d1}, [r4:16]!
	vst1.8	{d0, d1}, [r4:32]!
	vst1.8	{d0, d1}, [r4:64]!
	vst1.8	{d0, d1}, [r4:128]!
	vst1.8	{d0, d1}, [r4:256]!

@ CHECK: vst1.8 {d0, d1}, [r4]!         @ encoding: [0x04,0xf9,0x0d,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vst1.8 {d0, d1}, [r4:64]!      @ encoding: [0x04,0xf9,0x1d,0x0a]
@ CHECK: vst1.8 {d0, d1}, [r4:128]!     @ encoding: [0x04,0xf9,0x2d,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vst1.8	{d0, d1}, [r4], r6
	vst1.8	{d0, d1}, [r4:16], r6
	vst1.8	{d0, d1}, [r4:32], r6
	vst1.8	{d0, d1}, [r4:64], r6
	vst1.8	{d0, d1}, [r4:128], r6
	vst1.8	{d0, d1}, [r4:256], r6

@ CHECK: vst1.8 {d0, d1}, [r4], r6      @ encoding: [0x04,0xf9,0x06,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vst1.8 {d0, d1}, [r4:64], r6   @ encoding: [0x04,0xf9,0x16,0x0a]
@ CHECK: vst1.8 {d0, d1}, [r4:128], r6  @ encoding: [0x04,0xf9,0x26,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vst1.8	{d0, d1, d2}, [r4]
	vst1.8	{d0, d1, d2}, [r4:16]
	vst1.8	{d0, d1, d2}, [r4:32]
	vst1.8	{d0, d1, d2}, [r4:64]
	vst1.8	{d0, d1, d2}, [r4:128]
	vst1.8	{d0, d1, d2}, [r4:256]

@ CHECK: vst1.8 {d0, d1, d2}, [r4]      @ encoding: [0x04,0xf9,0x0f,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vst1.8 {d0, d1, d2}, [r4:64]   @ encoding: [0x04,0xf9,0x1f,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vst1.8	{d0, d1, d2}, [r4]!
	vst1.8	{d0, d1, d2}, [r4:16]!
	vst1.8	{d0, d1, d2}, [r4:32]!
	vst1.8	{d0, d1, d2}, [r4:64]!
	vst1.8	{d0, d1, d2}, [r4:128]!
	vst1.8	{d0, d1, d2}, [r4:256]!

@ CHECK: vst1.8 {d0, d1, d2}, [r4]!     @ encoding: [0x04,0xf9,0x0d,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vst1.8 {d0, d1, d2}, [r4:64]!  @ encoding: [0x04,0xf9,0x1d,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vst1.8	{d0, d1, d2}, [r4], r6
	vst1.8	{d0, d1, d2}, [r4:16], r6
	vst1.8	{d0, d1, d2}, [r4:32], r6
	vst1.8	{d0, d1, d2}, [r4:64], r6
	vst1.8	{d0, d1, d2}, [r4:128], r6
	vst1.8	{d0, d1, d2}, [r4:256], r6

@ CHECK: vst1.8 {d0, d1, d2}, [r4], r6  @ encoding: [0x04,0xf9,0x06,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vst1.8 {d0, d1, d2}, [r4:64], r6 @ encoding: [0x04,0xf9,0x16,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vst1.8	{d0, d1, d2, d3}, [r4]
	vst1.8	{d0, d1, d2, d3}, [r4:16]
	vst1.8	{d0, d1, d2, d3}, [r4:32]
	vst1.8	{d0, d1, d2, d3}, [r4:64]
	vst1.8	{d0, d1, d2, d3}, [r4:128]
	vst1.8	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vst1.8 {d0, d1, d2, d3}, [r4]  @ encoding: [0x04,0xf9,0x0f,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vst1.8 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x04,0xf9,0x1f,0x02]
@ CHECK: vst1.8 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x04,0xf9,0x2f,0x02]
@ CHECK: vst1.8 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x04,0xf9,0x3f,0x02]

	vst1.8	{d0, d1, d2, d3}, [r4]!
	vst1.8	{d0, d1, d2, d3}, [r4:16]!
	vst1.8	{d0, d1, d2, d3}, [r4:32]!
	vst1.8	{d0, d1, d2, d3}, [r4:64]!
	vst1.8	{d0, d1, d2, d3}, [r4:128]!
	vst1.8	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vst1.8 {d0, d1, d2, d3}, [r4]! @ encoding: [0x04,0xf9,0x0d,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vst1.8 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x04,0xf9,0x1d,0x02]
@ CHECK: vst1.8 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x04,0xf9,0x2d,0x02]
@ CHECK: vst1.8 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x04,0xf9,0x3d,0x02]

	vst1.8	{d0, d1, d2, d3}, [r4], r6
	vst1.8	{d0, d1, d2, d3}, [r4:16], r6
	vst1.8	{d0, d1, d2, d3}, [r4:32], r6
	vst1.8	{d0, d1, d2, d3}, [r4:64], r6
	vst1.8	{d0, d1, d2, d3}, [r4:128], r6
	vst1.8	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vst1.8 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x04,0xf9,0x06,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.8  {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vst1.8 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x04,0xf9,0x16,0x02]
@ CHECK: vst1.8 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x04,0xf9,0x26,0x02]
@ CHECK: vst1.8 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x04,0xf9,0x36,0x02]

	vst1.8	{d0[2]}, [r4]
	vst1.8	{d0[2]}, [r4:16]
	vst1.8	{d0[2]}, [r4:32]
	vst1.8	{d0[2]}, [r4:64]
	vst1.8	{d0[2]}, [r4:128]
	vst1.8	{d0[2]}, [r4:256]

@ CHECK: vst1.8 {d0[2]}, [r4]           @ encoding: [0x84,0xf9,0x4f,0x00]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:16]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:32]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:64]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:128]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:256]
@ CHECK-ERRORS:                              ^

	vst1.8	{d0[2]}, [r4]!
	vst1.8	{d0[2]}, [r4:16]!
	vst1.8	{d0[2]}, [r4:32]!
	vst1.8	{d0[2]}, [r4:64]!
	vst1.8	{d0[2]}, [r4:128]!
	vst1.8	{d0[2]}, [r4:256]!

@ CHECK: vst1.8 {d0[2]}, [r4]!          @ encoding: [0x84,0xf9,0x4d,0x00]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:16]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:32]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:64]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:128]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:256]!
@ CHECK-ERRORS:                              ^

	vst1.8	{d0[2]}, [r4], r6
	vst1.8	{d0[2]}, [r4:16], r6
	vst1.8	{d0[2]}, [r4:32], r6
	vst1.8	{d0[2]}, [r4:64], r6
	vst1.8	{d0[2]}, [r4:128], r6
	vst1.8	{d0[2]}, [r4:256], r6

@ CHECK: vst1.8 {d0[2]}, [r4], r6       @ encoding: [0x84,0xf9,0x46,0x00]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:16], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:32], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:64], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:128], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst1.8  {d0[2]}, [r4:256], r6
@ CHECK-ERRORS:                              ^

	vst1.16	{d0}, [r4]
	vst1.16	{d0}, [r4:16]
	vst1.16	{d0}, [r4:32]
	vst1.16	{d0}, [r4:64]
	vst1.16	{d0}, [r4:128]
	vst1.16	{d0}, [r4:256]

@ CHECK: vst1.16 {d0}, [r4]              @ encoding: [0x04,0xf9,0x4f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0}, [r4:16]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0}, [r4:32]
@ CHECK-ERRORS:                           ^
@ CHECK: vst1.16 {d0}, [r4:64]           @ encoding: [0x04,0xf9,0x5f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0}, [r4:128]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0}, [r4:256]
@ CHECK-ERRORS:                           ^

	vst1.16	{d0}, [r4]!
	vst1.16	{d0}, [r4:16]!
	vst1.16	{d0}, [r4:32]!
	vst1.16	{d0}, [r4:64]!
	vst1.16	{d0}, [r4:128]!
	vst1.16	{d0}, [r4:256]!

@ CHECK: vst1.16 {d0}, [r4]!             @ encoding: [0x04,0xf9,0x4d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0}, [r4:16]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0}, [r4:32]!
@ CHECK-ERRORS:                           ^
@ CHECK: vst1.16 {d0}, [r4:64]!          @ encoding: [0x04,0xf9,0x5d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0}, [r4:128]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0}, [r4:256]!
@ CHECK-ERRORS:                           ^

	vst1.16	{d0}, [r4], r6
	vst1.16	{d0}, [r4:16], r6
	vst1.16	{d0}, [r4:32], r6
	vst1.16	{d0}, [r4:64], r6
	vst1.16	{d0}, [r4:128], r6
	vst1.16	{d0}, [r4:256], r6

@ CHECK: vst1.16 {d0}, [r4], r6          @ encoding: [0x04,0xf9,0x46,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0}, [r4:16], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0}, [r4:32], r6
@ CHECK-ERRORS:                           ^
@ CHECK: vst1.16 {d0}, [r4:64], r6       @ encoding: [0x04,0xf9,0x56,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0}, [r4:128], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0}, [r4:256], r6
@ CHECK-ERRORS:                           ^

	vst1.16	{d0, d1}, [r4]
	vst1.16	{d0, d1}, [r4:16]
	vst1.16	{d0, d1}, [r4:32]
	vst1.16	{d0, d1}, [r4:64]
	vst1.16	{d0, d1}, [r4:128]
	vst1.16	{d0, d1}, [r4:256]

@ CHECK: vst1.16 {d0, d1}, [r4]          @ encoding: [0x04,0xf9,0x4f,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vst1.16 {d0, d1}, [r4:64]       @ encoding: [0x04,0xf9,0x5f,0x0a]
@ CHECK: vst1.16 {d0, d1}, [r4:128]      @ encoding: [0x04,0xf9,0x6f,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1}, [r4:256]
@ CHECK-ERRORS:                               ^

	vst1.16	{d0, d1}, [r4]!
	vst1.16	{d0, d1}, [r4:16]!
	vst1.16	{d0, d1}, [r4:32]!
	vst1.16	{d0, d1}, [r4:64]!
	vst1.16	{d0, d1}, [r4:128]!
	vst1.16	{d0, d1}, [r4:256]!

@ CHECK: vst1.16 {d0, d1}, [r4]!         @ encoding: [0x04,0xf9,0x4d,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vst1.16 {d0, d1}, [r4:64]!      @ encoding: [0x04,0xf9,0x5d,0x0a]
@ CHECK: vst1.16 {d0, d1}, [r4:128]!     @ encoding: [0x04,0xf9,0x6d,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vst1.16	{d0, d1}, [r4], r6
	vst1.16	{d0, d1}, [r4:16], r6
	vst1.16	{d0, d1}, [r4:32], r6
	vst1.16	{d0, d1}, [r4:64], r6
	vst1.16	{d0, d1}, [r4:128], r6
	vst1.16	{d0, d1}, [r4:256], r6

@ CHECK: vst1.16 {d0, d1}, [r4], r6      @ encoding: [0x04,0xf9,0x46,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vst1.16 {d0, d1}, [r4:64], r6   @ encoding: [0x04,0xf9,0x56,0x0a]
@ CHECK: vst1.16 {d0, d1}, [r4:128], r6  @ encoding: [0x04,0xf9,0x66,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vst1.16	{d0, d1, d2}, [r4]
	vst1.16	{d0, d1, d2}, [r4:16]
	vst1.16	{d0, d1, d2}, [r4:32]
	vst1.16	{d0, d1, d2}, [r4:64]
	vst1.16	{d0, d1, d2}, [r4:128]
	vst1.16	{d0, d1, d2}, [r4:256]

@ CHECK: vst1.16 {d0, d1, d2}, [r4]      @ encoding: [0x04,0xf9,0x4f,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vst1.16 {d0, d1, d2}, [r4:64]   @ encoding: [0x04,0xf9,0x5f,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vst1.16	{d0, d1, d2}, [r4]!
	vst1.16	{d0, d1, d2}, [r4:16]!
	vst1.16	{d0, d1, d2}, [r4:32]!
	vst1.16	{d0, d1, d2}, [r4:64]!
	vst1.16	{d0, d1, d2}, [r4:128]!
	vst1.16	{d0, d1, d2}, [r4:256]!

@ CHECK: vst1.16 {d0, d1, d2}, [r4]!     @ encoding: [0x04,0xf9,0x4d,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vst1.16 {d0, d1, d2}, [r4:64]!  @ encoding: [0x04,0xf9,0x5d,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vst1.16	{d0, d1, d2}, [r4], r6
	vst1.16	{d0, d1, d2}, [r4:16], r6
	vst1.16	{d0, d1, d2}, [r4:32], r6
	vst1.16	{d0, d1, d2}, [r4:64], r6
	vst1.16	{d0, d1, d2}, [r4:128], r6
	vst1.16	{d0, d1, d2}, [r4:256], r6

@ CHECK: vst1.16 {d0, d1, d2}, [r4], r6  @ encoding: [0x04,0xf9,0x46,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vst1.16 {d0, d1, d2}, [r4:64], r6 @ encoding: [0x04,0xf9,0x56,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vst1.16	{d0, d1, d2, d3}, [r4]
	vst1.16	{d0, d1, d2, d3}, [r4:16]
	vst1.16	{d0, d1, d2, d3}, [r4:32]
	vst1.16	{d0, d1, d2, d3}, [r4:64]
	vst1.16	{d0, d1, d2, d3}, [r4:128]
	vst1.16	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vst1.16 {d0, d1, d2, d3}, [r4]  @ encoding: [0x04,0xf9,0x4f,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vst1.16 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x04,0xf9,0x5f,0x02]
@ CHECK: vst1.16 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x04,0xf9,0x6f,0x02]
@ CHECK: vst1.16 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x04,0xf9,0x7f,0x02]

	vst1.16	{d0, d1, d2, d3}, [r4]!
	vst1.16	{d0, d1, d2, d3}, [r4:16]!
	vst1.16	{d0, d1, d2, d3}, [r4:32]!
	vst1.16	{d0, d1, d2, d3}, [r4:64]!
	vst1.16	{d0, d1, d2, d3}, [r4:128]!
	vst1.16	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vst1.16 {d0, d1, d2, d3}, [r4]! @ encoding: [0x04,0xf9,0x4d,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vst1.16 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x04,0xf9,0x5d,0x02]
@ CHECK: vst1.16 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x04,0xf9,0x6d,0x02]
@ CHECK: vst1.16 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x04,0xf9,0x7d,0x02]

	vst1.16	{d0, d1, d2, d3}, [r4], r6
	vst1.16	{d0, d1, d2, d3}, [r4:16], r6
	vst1.16	{d0, d1, d2, d3}, [r4:32], r6
	vst1.16	{d0, d1, d2, d3}, [r4:64], r6
	vst1.16	{d0, d1, d2, d3}, [r4:128], r6
	vst1.16	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vst1.16 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x04,0xf9,0x46,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.16 {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vst1.16 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x04,0xf9,0x56,0x02]
@ CHECK: vst1.16 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x04,0xf9,0x66,0x02]
@ CHECK: vst1.16 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x04,0xf9,0x76,0x02]

	vst1.16	{d0[2]}, [r4]
	vst1.16	{d0[2]}, [r4:16]
	vst1.16	{d0[2]}, [r4:32]
	vst1.16	{d0[2]}, [r4:64]
	vst1.16	{d0[2]}, [r4:128]
	vst1.16	{d0[2]}, [r4:256]

@ CHECK: vst1.16 {d0[2]}, [r4]           @ encoding: [0x84,0xf9,0x8f,0x04]
@ CHECK: vst1.16 {d0[2]}, [r4:16]        @ encoding: [0x84,0xf9,0x9f,0x04]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst1.16 {d0[2]}, [r4:32]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst1.16 {d0[2]}, [r4:64]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst1.16 {d0[2]}, [r4:128]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst1.16 {d0[2]}, [r4:256]
@ CHECK-ERRORS:                              ^

	vst1.16	{d0[2]}, [r4]!
	vst1.16	{d0[2]}, [r4:16]!
	vst1.16	{d0[2]}, [r4:32]!
	vst1.16	{d0[2]}, [r4:64]!
	vst1.16	{d0[2]}, [r4:128]!
	vst1.16	{d0[2]}, [r4:256]!

@ CHECK: vst1.16 {d0[2]}, [r4]!          @ encoding: [0x84,0xf9,0x8d,0x04]
@ CHECK: vst1.16 {d0[2]}, [r4:16]!       @ encoding: [0x84,0xf9,0x9d,0x04]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst1.16 {d0[2]}, [r4:32]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst1.16 {d0[2]}, [r4:64]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst1.16 {d0[2]}, [r4:128]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst1.16 {d0[2]}, [r4:256]!
@ CHECK-ERRORS:                              ^

	vst1.16	{d0[2]}, [r4], r6
	vst1.16	{d0[2]}, [r4:16], r6
	vst1.16	{d0[2]}, [r4:32], r6
	vst1.16	{d0[2]}, [r4:64], r6
	vst1.16	{d0[2]}, [r4:128], r6
	vst1.16	{d0[2]}, [r4:256], r6

@ CHECK: vst1.16 {d0[2]}, [r4], r6       @ encoding: [0x84,0xf9,0x86,0x04]
@ CHECK: vst1.16 {d0[2]}, [r4:16], r6    @ encoding: [0x84,0xf9,0x96,0x04]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst1.16 {d0[2]}, [r4:32], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst1.16 {d0[2]}, [r4:64], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst1.16 {d0[2]}, [r4:128], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst1.16 {d0[2]}, [r4:256], r6
@ CHECK-ERRORS:                              ^

	vst1.32	{d0}, [r4]
	vst1.32	{d0}, [r4:16]
	vst1.32	{d0}, [r4:32]
	vst1.32	{d0}, [r4:64]
	vst1.32	{d0}, [r4:128]
	vst1.32	{d0}, [r4:256]

@ CHECK: vst1.32 {d0}, [r4]              @ encoding: [0x04,0xf9,0x8f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0}, [r4:16]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0}, [r4:32]
@ CHECK-ERRORS:                           ^
@ CHECK: vst1.32 {d0}, [r4:64]           @ encoding: [0x04,0xf9,0x9f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0}, [r4:128]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0}, [r4:256]
@ CHECK-ERRORS:                           ^

	vst1.32	{d0}, [r4]!
	vst1.32	{d0}, [r4:16]!
	vst1.32	{d0}, [r4:32]!
	vst1.32	{d0}, [r4:64]!
	vst1.32	{d0}, [r4:128]!
	vst1.32	{d0}, [r4:256]!

@ CHECK: vst1.32 {d0}, [r4]!             @ encoding: [0x04,0xf9,0x8d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0}, [r4:16]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0}, [r4:32]!
@ CHECK-ERRORS:                           ^
@ CHECK: vst1.32 {d0}, [r4:64]!          @ encoding: [0x04,0xf9,0x9d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0}, [r4:128]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0}, [r4:256]!
@ CHECK-ERRORS:                           ^

	vst1.32	{d0}, [r4], r6
	vst1.32	{d0}, [r4:16], r6
	vst1.32	{d0}, [r4:32], r6
	vst1.32	{d0}, [r4:64], r6
	vst1.32	{d0}, [r4:128], r6
	vst1.32	{d0}, [r4:256], r6

@ CHECK: vst1.32 {d0}, [r4], r6          @ encoding: [0x04,0xf9,0x86,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0}, [r4:16], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0}, [r4:32], r6
@ CHECK-ERRORS:                           ^
@ CHECK: vst1.32 {d0}, [r4:64], r6       @ encoding: [0x04,0xf9,0x96,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0}, [r4:128], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0}, [r4:256], r6
@ CHECK-ERRORS:                           ^

	vst1.32	{d0, d1}, [r4]
	vst1.32	{d0, d1}, [r4:16]
	vst1.32	{d0, d1}, [r4:32]
	vst1.32	{d0, d1}, [r4:64]
	vst1.32	{d0, d1}, [r4:128]
	vst1.32	{d0, d1}, [r4:256]

@ CHECK: vst1.32 {d0, d1}, [r4]          @ encoding: [0x04,0xf9,0x8f,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vst1.32 {d0, d1}, [r4:64]       @ encoding: [0x04,0xf9,0x9f,0x0a]
@ CHECK: vst1.32 {d0, d1}, [r4:128]      @ encoding: [0x04,0xf9,0xaf,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1}, [r4:256]
@ CHECK-ERRORS:                               ^

	vst1.32	{d0, d1}, [r4]!
	vst1.32	{d0, d1}, [r4:16]!
	vst1.32	{d0, d1}, [r4:32]!
	vst1.32	{d0, d1}, [r4:64]!
	vst1.32	{d0, d1}, [r4:128]!
	vst1.32	{d0, d1}, [r4:256]!

@ CHECK: vst1.32 {d0, d1}, [r4]!         @ encoding: [0x04,0xf9,0x8d,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vst1.32 {d0, d1}, [r4:64]!      @ encoding: [0x04,0xf9,0x9d,0x0a]
@ CHECK: vst1.32 {d0, d1}, [r4:128]!     @ encoding: [0x04,0xf9,0xad,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vst1.32	{d0, d1}, [r4], r6
	vst1.32	{d0, d1}, [r4:16], r6
	vst1.32	{d0, d1}, [r4:32], r6
	vst1.32	{d0, d1}, [r4:64], r6
	vst1.32	{d0, d1}, [r4:128], r6
	vst1.32	{d0, d1}, [r4:256], r6

@ CHECK: vst1.32 {d0, d1}, [r4], r6      @ encoding: [0x04,0xf9,0x86,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vst1.32 {d0, d1}, [r4:64], r6   @ encoding: [0x04,0xf9,0x96,0x0a]
@ CHECK: vst1.32 {d0, d1}, [r4:128], r6  @ encoding: [0x04,0xf9,0xa6,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vst1.32	{d0, d1, d2}, [r4]
	vst1.32	{d0, d1, d2}, [r4:16]
	vst1.32	{d0, d1, d2}, [r4:32]
	vst1.32	{d0, d1, d2}, [r4:64]
	vst1.32	{d0, d1, d2}, [r4:128]
	vst1.32	{d0, d1, d2}, [r4:256]

@ CHECK: vst1.32 {d0, d1, d2}, [r4]      @ encoding: [0x04,0xf9,0x8f,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vst1.32 {d0, d1, d2}, [r4:64]   @ encoding: [0x04,0xf9,0x9f,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vst1.32	{d0, d1, d2}, [r4]!
	vst1.32	{d0, d1, d2}, [r4:16]!
	vst1.32	{d0, d1, d2}, [r4:32]!
	vst1.32	{d0, d1, d2}, [r4:64]!
	vst1.32	{d0, d1, d2}, [r4:128]!
	vst1.32	{d0, d1, d2}, [r4:256]!

@ CHECK: vst1.32 {d0, d1, d2}, [r4]!     @ encoding: [0x04,0xf9,0x8d,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vst1.32 {d0, d1, d2}, [r4:64]!  @ encoding: [0x04,0xf9,0x9d,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vst1.32	{d0, d1, d2}, [r4], r6
	vst1.32	{d0, d1, d2}, [r4:16], r6
	vst1.32	{d0, d1, d2}, [r4:32], r6
	vst1.32	{d0, d1, d2}, [r4:64], r6
	vst1.32	{d0, d1, d2}, [r4:128], r6
	vst1.32	{d0, d1, d2}, [r4:256], r6

@ CHECK: vst1.32 {d0, d1, d2}, [r4], r6  @ encoding: [0x04,0xf9,0x86,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vst1.32 {d0, d1, d2}, [r4:64], r6 @ encoding: [0x04,0xf9,0x96,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vst1.32	{d0, d1, d2, d3}, [r4]
	vst1.32	{d0, d1, d2, d3}, [r4:16]
	vst1.32	{d0, d1, d2, d3}, [r4:32]
	vst1.32	{d0, d1, d2, d3}, [r4:64]
	vst1.32	{d0, d1, d2, d3}, [r4:128]
	vst1.32	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vst1.32 {d0, d1, d2, d3}, [r4]  @ encoding: [0x04,0xf9,0x8f,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vst1.32 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x04,0xf9,0x9f,0x02]
@ CHECK: vst1.32 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x04,0xf9,0xaf,0x02]
@ CHECK: vst1.32 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x04,0xf9,0xbf,0x02]

	vst1.32	{d0, d1, d2, d3}, [r4]!
	vst1.32	{d0, d1, d2, d3}, [r4:16]!
	vst1.32	{d0, d1, d2, d3}, [r4:32]!
	vst1.32	{d0, d1, d2, d3}, [r4:64]!
	vst1.32	{d0, d1, d2, d3}, [r4:128]!
	vst1.32	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vst1.32 {d0, d1, d2, d3}, [r4]! @ encoding: [0x04,0xf9,0x8d,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vst1.32 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x04,0xf9,0x9d,0x02]
@ CHECK: vst1.32 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x04,0xf9,0xad,0x02]
@ CHECK: vst1.32 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x04,0xf9,0xbd,0x02]

	vst1.32	{d0, d1, d2, d3}, [r4], r6
	vst1.32	{d0, d1, d2, d3}, [r4:16], r6
	vst1.32	{d0, d1, d2, d3}, [r4:32], r6
	vst1.32	{d0, d1, d2, d3}, [r4:64], r6
	vst1.32	{d0, d1, d2, d3}, [r4:128], r6
	vst1.32	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vst1.32 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x04,0xf9,0x86,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.32 {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vst1.32 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x04,0xf9,0x96,0x02]
@ CHECK: vst1.32 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x04,0xf9,0xa6,0x02]
@ CHECK: vst1.32 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x04,0xf9,0xb6,0x02]

	vst1.32	{d0[1]}, [r4]
	vst1.32	{d0[1]}, [r4:16]
	vst1.32	{d0[1]}, [r4:32]
	vst1.32	{d0[1]}, [r4:64]
	vst1.32	{d0[1]}, [r4:128]
	vst1.32	{d0[1]}, [r4:256]

@ CHECK: vst1.32 {d0[1]}, [r4]           @ encoding: [0x84,0xf9,0x8f,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst1.32 {d0[1]}, [r4:16]
@ CHECK-ERRORS:                              ^
@ CHECK: vst1.32 {d0[1]}, [r4:32]        @ encoding: [0x84,0xf9,0xbf,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst1.32 {d0[1]}, [r4:64]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst1.32 {d0[1]}, [r4:128]
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst1.32 {d0[1]}, [r4:256]
@ CHECK-ERRORS:                              ^

	vst1.32	{d0[1]}, [r4]!
	vst1.32	{d0[1]}, [r4:16]!
	vst1.32	{d0[1]}, [r4:32]!
	vst1.32	{d0[1]}, [r4:64]!
	vst1.32	{d0[1]}, [r4:128]!
	vst1.32	{d0[1]}, [r4:256]!

@ CHECK: vst1.32 {d0[1]}, [r4]!          @ encoding: [0x84,0xf9,0x8d,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst1.32 {d0[1]}, [r4:16]!
@ CHECK-ERRORS:                              ^
@ CHECK: vst1.32 {d0[1]}, [r4:32]!       @ encoding: [0x84,0xf9,0xbd,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst1.32 {d0[1]}, [r4:64]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst1.32 {d0[1]}, [r4:128]!
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst1.32 {d0[1]}, [r4:256]!
@ CHECK-ERRORS:                              ^

	vst1.32	{d0[1]}, [r4], r6
	vst1.32	{d0[1]}, [r4:16], r6
	vst1.32	{d0[1]}, [r4:32], r6
	vst1.32	{d0[1]}, [r4:64], r6
	vst1.32	{d0[1]}, [r4:128], r6
	vst1.32	{d0[1]}, [r4:256], r6

@ CHECK: vst1.32 {d0[1]}, [r4], r6       @ encoding: [0x84,0xf9,0x86,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst1.32 {d0[1]}, [r4:16], r6
@ CHECK-ERRORS:                              ^
@ CHECK: vst1.32 {d0[1]}, [r4:32], r6    @ encoding: [0x84,0xf9,0xb6,0x08]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst1.32 {d0[1]}, [r4:64], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst1.32 {d0[1]}, [r4:128], r6
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst1.32 {d0[1]}, [r4:256], r6
@ CHECK-ERRORS:                              ^

	vst1.64	{d0}, [r4]
	vst1.64	{d0}, [r4:16]
	vst1.64	{d0}, [r4:32]
	vst1.64	{d0}, [r4:64]
	vst1.64	{d0}, [r4:128]
	vst1.64	{d0}, [r4:256]

@ CHECK: vst1.64 {d0}, [r4]              @ encoding: [0x04,0xf9,0xcf,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0}, [r4:16]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0}, [r4:32]
@ CHECK-ERRORS:                           ^
@ CHECK: vst1.64 {d0}, [r4:64]           @ encoding: [0x04,0xf9,0xdf,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0}, [r4:128]
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0}, [r4:256]
@ CHECK-ERRORS:                           ^

	vst1.64	{d0}, [r4]!
	vst1.64	{d0}, [r4:16]!
	vst1.64	{d0}, [r4:32]!
	vst1.64	{d0}, [r4:64]!
	vst1.64	{d0}, [r4:128]!
	vst1.64	{d0}, [r4:256]!

@ CHECK: vst1.64 {d0}, [r4]!             @ encoding: [0x04,0xf9,0xcd,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0}, [r4:16]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0}, [r4:32]!
@ CHECK-ERRORS:                           ^
@ CHECK: vst1.64 {d0}, [r4:64]!          @ encoding: [0x04,0xf9,0xdd,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0}, [r4:128]!
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0}, [r4:256]!
@ CHECK-ERRORS:                           ^

	vst1.64	{d0}, [r4], r6
	vst1.64	{d0}, [r4:16], r6
	vst1.64	{d0}, [r4:32], r6
	vst1.64	{d0}, [r4:64], r6
	vst1.64	{d0}, [r4:128], r6
	vst1.64	{d0}, [r4:256], r6

@ CHECK: vst1.64 {d0}, [r4], r6          @ encoding: [0x04,0xf9,0xc6,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0}, [r4:16], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0}, [r4:32], r6
@ CHECK-ERRORS:                           ^
@ CHECK: vst1.64 {d0}, [r4:64], r6       @ encoding: [0x04,0xf9,0xd6,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0}, [r4:128], r6
@ CHECK-ERRORS:                           ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0}, [r4:256], r6
@ CHECK-ERRORS:                           ^

	vst1.64	{d0, d1}, [r4]
	vst1.64	{d0, d1}, [r4:16]
	vst1.64	{d0, d1}, [r4:32]
	vst1.64	{d0, d1}, [r4:64]
	vst1.64	{d0, d1}, [r4:128]
	vst1.64	{d0, d1}, [r4:256]

@ CHECK: vst1.64 {d0, d1}, [r4]          @ encoding: [0x04,0xf9,0xcf,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vst1.64 {d0, d1}, [r4:64]       @ encoding: [0x04,0xf9,0xdf,0x0a]
@ CHECK: vst1.64 {d0, d1}, [r4:128]      @ encoding: [0x04,0xf9,0xef,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1}, [r4:256]
@ CHECK-ERRORS:                               ^

	vst1.64	{d0, d1}, [r4]!
	vst1.64	{d0, d1}, [r4:16]!
	vst1.64	{d0, d1}, [r4:32]!
	vst1.64	{d0, d1}, [r4:64]!
	vst1.64	{d0, d1}, [r4:128]!
	vst1.64	{d0, d1}, [r4:256]!

@ CHECK: vst1.64 {d0, d1}, [r4]!         @ encoding: [0x04,0xf9,0xcd,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vst1.64 {d0, d1}, [r4:64]!      @ encoding: [0x04,0xf9,0xdd,0x0a]
@ CHECK: vst1.64 {d0, d1}, [r4:128]!     @ encoding: [0x04,0xf9,0xed,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vst1.64	{d0, d1}, [r4], r6
	vst1.64	{d0, d1}, [r4:16], r6
	vst1.64	{d0, d1}, [r4:32], r6
	vst1.64	{d0, d1}, [r4:64], r6
	vst1.64	{d0, d1}, [r4:128], r6
	vst1.64	{d0, d1}, [r4:256], r6

@ CHECK: vst1.64 {d0, d1}, [r4], r6      @ encoding: [0x04,0xf9,0xc6,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vst1.64 {d0, d1}, [r4:64], r6   @ encoding: [0x04,0xf9,0xd6,0x0a]
@ CHECK: vst1.64 {d0, d1}, [r4:128], r6  @ encoding: [0x04,0xf9,0xe6,0x0a]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vst1.64	{d0, d1, d2}, [r4]
	vst1.64	{d0, d1, d2}, [r4:16]
	vst1.64	{d0, d1, d2}, [r4:32]
	vst1.64	{d0, d1, d2}, [r4:64]
	vst1.64	{d0, d1, d2}, [r4:128]
	vst1.64	{d0, d1, d2}, [r4:256]

@ CHECK: vst1.64 {d0, d1, d2}, [r4]      @ encoding: [0x04,0xf9,0xcf,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vst1.64 {d0, d1, d2}, [r4:64]   @ encoding: [0x04,0xf9,0xdf,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vst1.64	{d0, d1, d2}, [r4]!
	vst1.64	{d0, d1, d2}, [r4:16]!
	vst1.64	{d0, d1, d2}, [r4:32]!
	vst1.64	{d0, d1, d2}, [r4:64]!
	vst1.64	{d0, d1, d2}, [r4:128]!
	vst1.64	{d0, d1, d2}, [r4:256]!

@ CHECK: vst1.64 {d0, d1, d2}, [r4]!     @ encoding: [0x04,0xf9,0xcd,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vst1.64 {d0, d1, d2}, [r4:64]!  @ encoding: [0x04,0xf9,0xdd,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vst1.64	{d0, d1, d2}, [r4], r6
	vst1.64	{d0, d1, d2}, [r4:16], r6
	vst1.64	{d0, d1, d2}, [r4:32], r6
	vst1.64	{d0, d1, d2}, [r4:64], r6
	vst1.64	{d0, d1, d2}, [r4:128], r6
	vst1.64	{d0, d1, d2}, [r4:256], r6

@ CHECK: vst1.64 {d0, d1, d2}, [r4], r6  @ encoding: [0x04,0xf9,0xc6,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vst1.64 {d0, d1, d2}, [r4:64], r6 @ encoding: [0x04,0xf9,0xd6,0x06]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vst1.64	{d0, d1, d2, d3}, [r4]
	vst1.64	{d0, d1, d2, d3}, [r4:16]
	vst1.64	{d0, d1, d2, d3}, [r4:32]
	vst1.64	{d0, d1, d2, d3}, [r4:64]
	vst1.64	{d0, d1, d2, d3}, [r4:128]
	vst1.64	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vst1.64 {d0, d1, d2, d3}, [r4]  @ encoding: [0x04,0xf9,0xcf,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vst1.64 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x04,0xf9,0xdf,0x02]
@ CHECK: vst1.64 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x04,0xf9,0xef,0x02]
@ CHECK: vst1.64 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x04,0xf9,0xff,0x02]

	vst1.64	{d0, d1, d2, d3}, [r4]!
	vst1.64	{d0, d1, d2, d3}, [r4:16]!
	vst1.64	{d0, d1, d2, d3}, [r4:32]!
	vst1.64	{d0, d1, d2, d3}, [r4:64]!
	vst1.64	{d0, d1, d2, d3}, [r4:128]!
	vst1.64	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vst1.64 {d0, d1, d2, d3}, [r4]! @ encoding: [0x04,0xf9,0xcd,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vst1.64 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x04,0xf9,0xdd,0x02]
@ CHECK: vst1.64 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x04,0xf9,0xed,0x02]
@ CHECK: vst1.64 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x04,0xf9,0xfd,0x02]

	vst1.64	{d0, d1, d2, d3}, [r4], r6
	vst1.64	{d0, d1, d2, d3}, [r4:16], r6
	vst1.64	{d0, d1, d2, d3}, [r4:32], r6
	vst1.64	{d0, d1, d2, d3}, [r4:64], r6
	vst1.64	{d0, d1, d2, d3}, [r4:128], r6
	vst1.64	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vst1.64 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x04,0xf9,0xc6,0x02]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst1.64 {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vst1.64 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x04,0xf9,0xd6,0x02]
@ CHECK: vst1.64 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x04,0xf9,0xe6,0x02]
@ CHECK: vst1.64 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x04,0xf9,0xf6,0x02]

	vst2.8	{d0, d1}, [r4]
	vst2.8	{d0, d1}, [r4:16]
	vst2.8	{d0, d1}, [r4:32]
	vst2.8	{d0, d1}, [r4:64]
	vst2.8	{d0, d1}, [r4:128]
	vst2.8	{d0, d1}, [r4:256]

@ CHECK: vst2.8 {d0, d1}, [r4]          @ encoding: [0x04,0xf9,0x0f,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vst2.8 {d0, d1}, [r4:64]       @ encoding: [0x04,0xf9,0x1f,0x08]
@ CHECK: vst2.8 {d0, d1}, [r4:128]      @ encoding: [0x04,0xf9,0x2f,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1}, [r4:256]
@ CHECK-ERRORS:                               ^

	vst2.8	{d0, d1}, [r4]!
	vst2.8	{d0, d1}, [r4:16]!
	vst2.8	{d0, d1}, [r4:32]!
	vst2.8	{d0, d1}, [r4:64]!
	vst2.8	{d0, d1}, [r4:128]!
	vst2.8	{d0, d1}, [r4:256]!

@ CHECK: vst2.8 {d0, d1}, [r4]!         @ encoding: [0x04,0xf9,0x0d,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vst2.8 {d0, d1}, [r4:64]!      @ encoding: [0x04,0xf9,0x1d,0x08]
@ CHECK: vst2.8 {d0, d1}, [r4:128]!     @ encoding: [0x04,0xf9,0x2d,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vst2.8	{d0, d1}, [r4], r6
	vst2.8	{d0, d1}, [r4:16], r6
	vst2.8	{d0, d1}, [r4:32], r6
	vst2.8	{d0, d1}, [r4:64], r6
	vst2.8	{d0, d1}, [r4:128], r6
	vst2.8	{d0, d1}, [r4:256], r6

@ CHECK: vst2.8 {d0, d1}, [r4], r6      @ encoding: [0x04,0xf9,0x06,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vst2.8 {d0, d1}, [r4:64], r6   @ encoding: [0x04,0xf9,0x16,0x08]
@ CHECK: vst2.8 {d0, d1}, [r4:128], r6  @ encoding: [0x04,0xf9,0x26,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vst2.8	{d0, d2}, [r4]
	vst2.8	{d0, d2}, [r4:16]
	vst2.8	{d0, d2}, [r4:32]
	vst2.8	{d0, d2}, [r4:64]
	vst2.8	{d0, d2}, [r4:128]
	vst2.8	{d0, d2}, [r4:256]

@ CHECK: vst2.8 {d0, d2}, [r4]          @ encoding: [0x04,0xf9,0x0f,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d2}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d2}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vst2.8 {d0, d2}, [r4:64]       @ encoding: [0x04,0xf9,0x1f,0x09]
@ CHECK: vst2.8 {d0, d2}, [r4:128]      @ encoding: [0x04,0xf9,0x2f,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d2}, [r4:256]
@ CHECK-ERRORS:                               ^

	vst2.8	{d0, d2}, [r4]!
	vst2.8	{d0, d2}, [r4:16]!
	vst2.8	{d0, d2}, [r4:32]!
	vst2.8	{d0, d2}, [r4:64]!
	vst2.8	{d0, d2}, [r4:128]!
	vst2.8	{d0, d2}, [r4:256]!

@ CHECK: vst2.8 {d0, d2}, [r4]!         @ encoding: [0x04,0xf9,0x0d,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d2}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d2}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vst2.8 {d0, d2}, [r4:64]!      @ encoding: [0x04,0xf9,0x1d,0x09]
@ CHECK: vst2.8 {d0, d2}, [r4:128]!     @ encoding: [0x04,0xf9,0x2d,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d2}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vst2.8	{d0, d2}, [r4], r6
	vst2.8	{d0, d2}, [r4:16], r6
	vst2.8	{d0, d2}, [r4:32], r6
	vst2.8	{d0, d2}, [r4:64], r6
	vst2.8	{d0, d2}, [r4:128], r6
	vst2.8	{d0, d2}, [r4:256], r6

@ CHECK: vst2.8 {d0, d2}, [r4], r6      @ encoding: [0x04,0xf9,0x06,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d2}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d2}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vst2.8 {d0, d2}, [r4:64], r6   @ encoding: [0x04,0xf9,0x16,0x09]
@ CHECK: vst2.8 {d0, d2}, [r4:128], r6  @ encoding: [0x04,0xf9,0x26,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d2}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vst2.8	{d0, d1, d2, d3}, [r4]
	vst2.8	{d0, d1, d2, d3}, [r4:16]
	vst2.8	{d0, d1, d2, d3}, [r4:32]
	vst2.8	{d0, d1, d2, d3}, [r4:64]
	vst2.8	{d0, d1, d2, d3}, [r4:128]
	vst2.8	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vst2.8 {d0, d1, d2, d3}, [r4]  @ encoding: [0x04,0xf9,0x0f,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vst2.8 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x04,0xf9,0x1f,0x03]
@ CHECK: vst2.8 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x04,0xf9,0x2f,0x03]
@ CHECK: vst2.8 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x04,0xf9,0x3f,0x03]

	vst2.8	{d0, d1, d2, d3}, [r4]!
	vst2.8	{d0, d1, d2, d3}, [r4:16]!
	vst2.8	{d0, d1, d2, d3}, [r4:32]!
	vst2.8	{d0, d1, d2, d3}, [r4:64]!
	vst2.8	{d0, d1, d2, d3}, [r4:128]!
	vst2.8	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vst2.8 {d0, d1, d2, d3}, [r4]! @ encoding: [0x04,0xf9,0x0d,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vst2.8 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x04,0xf9,0x1d,0x03]
@ CHECK: vst2.8 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x04,0xf9,0x2d,0x03]
@ CHECK: vst2.8 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x04,0xf9,0x3d,0x03]

	vst2.8	{d0, d1, d2, d3}, [r4], r6
	vst2.8	{d0, d1, d2, d3}, [r4:16], r6
	vst2.8	{d0, d1, d2, d3}, [r4:32], r6
	vst2.8	{d0, d1, d2, d3}, [r4:64], r6
	vst2.8	{d0, d1, d2, d3}, [r4:128], r6
	vst2.8	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vst2.8 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x04,0xf9,0x06,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst2.8  {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vst2.8 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x04,0xf9,0x16,0x03]
@ CHECK: vst2.8 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x04,0xf9,0x26,0x03]
@ CHECK: vst2.8 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x04,0xf9,0x36,0x03]

	vst2.8	{d0[2], d1[2]}, [r4]
	vst2.8	{d0[2], d1[2]}, [r4:16]
	vst2.8	{d0[2], d1[2]}, [r4:32]
	vst2.8	{d0[2], d1[2]}, [r4:64]
	vst2.8	{d0[2], d1[2]}, [r4:128]
	vst2.8	{d0[2], d1[2]}, [r4:256]

@ CHECK: vst2.8 {d0[2], d1[2]}, [r4]    @ encoding: [0x84,0xf9,0x4f,0x01]
@ CHECK: vst2.8 {d0[2], d1[2]}, [r4:16] @ encoding: [0x84,0xf9,0x5f,0x01]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst2.8  {d0[2], d1[2]}, [r4:32]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst2.8  {d0[2], d1[2]}, [r4:64]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst2.8  {d0[2], d1[2]}, [r4:128]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst2.8  {d0[2], d1[2]}, [r4:256]
@ CHECK-ERRORS:                                     ^

	vst2.8	{d0[2], d1[2]}, [r4]!
	vst2.8	{d0[2], d1[2]}, [r4:16]!
	vst2.8	{d0[2], d1[2]}, [r4:32]!
	vst2.8	{d0[2], d1[2]}, [r4:64]!
	vst2.8	{d0[2], d1[2]}, [r4:128]!
	vst2.8	{d0[2], d1[2]}, [r4:256]!

@ CHECK: vst2.8 {d0[2], d1[2]}, [r4]!   @ encoding: [0x84,0xf9,0x4d,0x01]
@ CHECK: vst2.8 {d0[2], d1[2]}, [r4:16]! @ encoding: [0x84,0xf9,0x5d,0x01]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst2.8  {d0[2], d1[2]}, [r4:32]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst2.8  {d0[2], d1[2]}, [r4:64]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst2.8  {d0[2], d1[2]}, [r4:128]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst2.8  {d0[2], d1[2]}, [r4:256]!
@ CHECK-ERRORS:                                     ^

	vst2.8	{d0[2], d1[2]}, [r4], r6
	vst2.8	{d0[2], d1[2]}, [r4:16], r6
	vst2.8	{d0[2], d1[2]}, [r4:32], r6
	vst2.8	{d0[2], d1[2]}, [r4:64], r6
	vst2.8	{d0[2], d1[2]}, [r4:128], r6
	vst2.8	{d0[2], d1[2]}, [r4:256], r6

@ CHECK: vst2.8 {d0[2], d1[2]}, [r4], r6 @ encoding: [0x84,0xf9,0x46,0x01]
@ CHECK: vst2.8 {d0[2], d1[2]}, [r4:16], r6 @ encoding: [0x84,0xf9,0x56,0x01]
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst2.8  {d0[2], d1[2]}, [r4:32], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst2.8  {d0[2], d1[2]}, [r4:64], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst2.8  {d0[2], d1[2]}, [r4:128], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 16 or omitted
@ CHECK-ERRORS:         vst2.8  {d0[2], d1[2]}, [r4:256], r6
@ CHECK-ERRORS:                                     ^

	vst2.32	{d0, d1}, [r4]
	vst2.32	{d0, d1}, [r4:16]
	vst2.32	{d0, d1}, [r4:32]
	vst2.32	{d0, d1}, [r4:64]
	vst2.32	{d0, d1}, [r4:128]
	vst2.32	{d0, d1}, [r4:256]

@ CHECK: vst2.32 {d0, d1}, [r4]          @ encoding: [0x04,0xf9,0x8f,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vst2.32 {d0, d1}, [r4:64]       @ encoding: [0x04,0xf9,0x9f,0x08]
@ CHECK: vst2.32 {d0, d1}, [r4:128]      @ encoding: [0x04,0xf9,0xaf,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1}, [r4:256]
@ CHECK-ERRORS:                               ^

	vst2.32	{d0, d1}, [r4]!
	vst2.32	{d0, d1}, [r4:16]!
	vst2.32	{d0, d1}, [r4:32]!
	vst2.32	{d0, d1}, [r4:64]!
	vst2.32	{d0, d1}, [r4:128]!
	vst2.32	{d0, d1}, [r4:256]!

@ CHECK: vst2.32 {d0, d1}, [r4]!         @ encoding: [0x04,0xf9,0x8d,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vst2.32 {d0, d1}, [r4:64]!      @ encoding: [0x04,0xf9,0x9d,0x08]
@ CHECK: vst2.32 {d0, d1}, [r4:128]!     @ encoding: [0x04,0xf9,0xad,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vst2.32	{d0, d1}, [r4], r6
	vst2.32	{d0, d1}, [r4:16], r6
	vst2.32	{d0, d1}, [r4:32], r6
	vst2.32	{d0, d1}, [r4:64], r6
	vst2.32	{d0, d1}, [r4:128], r6
	vst2.32	{d0, d1}, [r4:256], r6

@ CHECK: vst2.32 {d0, d1}, [r4], r6      @ encoding: [0x04,0xf9,0x86,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vst2.32 {d0, d1}, [r4:64], r6   @ encoding: [0x04,0xf9,0x96,0x08]
@ CHECK: vst2.32 {d0, d1}, [r4:128], r6  @ encoding: [0x04,0xf9,0xa6,0x08]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vst2.32	{d0, d2}, [r4]
	vst2.32	{d0, d2}, [r4:16]
	vst2.32	{d0, d2}, [r4:32]
	vst2.32	{d0, d2}, [r4:64]
	vst2.32	{d0, d2}, [r4:128]
	vst2.32	{d0, d2}, [r4:256]

@ CHECK: vst2.32 {d0, d2}, [r4]          @ encoding: [0x04,0xf9,0x8f,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d2}, [r4:16]
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d2}, [r4:32]
@ CHECK-ERRORS:                               ^
@ CHECK: vst2.32 {d0, d2}, [r4:64]       @ encoding: [0x04,0xf9,0x9f,0x09]
@ CHECK: vst2.32 {d0, d2}, [r4:128]      @ encoding: [0x04,0xf9,0xaf,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d2}, [r4:256]
@ CHECK-ERRORS:                               ^

	vst2.32	{d0, d2}, [r4]!
	vst2.32	{d0, d2}, [r4:16]!
	vst2.32	{d0, d2}, [r4:32]!
	vst2.32	{d0, d2}, [r4:64]!
	vst2.32	{d0, d2}, [r4:128]!
	vst2.32	{d0, d2}, [r4:256]!

@ CHECK: vst2.32 {d0, d2}, [r4]!         @ encoding: [0x04,0xf9,0x8d,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d2}, [r4:16]!
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d2}, [r4:32]!
@ CHECK-ERRORS:                               ^
@ CHECK: vst2.32 {d0, d2}, [r4:64]!      @ encoding: [0x04,0xf9,0x9d,0x09]
@ CHECK: vst2.32 {d0, d2}, [r4:128]!     @ encoding: [0x04,0xf9,0xad,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d2}, [r4:256]!
@ CHECK-ERRORS:                               ^

	vst2.32	{d0, d2}, [r4], r6
	vst2.32	{d0, d2}, [r4:16], r6
	vst2.32	{d0, d2}, [r4:32], r6
	vst2.32	{d0, d2}, [r4:64], r6
	vst2.32	{d0, d2}, [r4:128], r6
	vst2.32	{d0, d2}, [r4:256], r6

@ CHECK: vst2.32 {d0, d2}, [r4], r6      @ encoding: [0x04,0xf9,0x86,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d2}, [r4:16], r6
@ CHECK-ERRORS:                               ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d2}, [r4:32], r6
@ CHECK-ERRORS:                               ^
@ CHECK: vst2.32 {d0, d2}, [r4:64], r6   @ encoding: [0x04,0xf9,0x96,0x09]
@ CHECK: vst2.32 {d0, d2}, [r4:128], r6  @ encoding: [0x04,0xf9,0xa6,0x09]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d2}, [r4:256], r6
@ CHECK-ERRORS:                               ^

	vst2.32	{d0, d1, d2, d3}, [r4]
	vst2.32	{d0, d1, d2, d3}, [r4:16]
	vst2.32	{d0, d1, d2, d3}, [r4:32]
	vst2.32	{d0, d1, d2, d3}, [r4:64]
	vst2.32	{d0, d1, d2, d3}, [r4:128]
	vst2.32	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vst2.32 {d0, d1, d2, d3}, [r4]  @ encoding: [0x04,0xf9,0x8f,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vst2.32 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x04,0xf9,0x9f,0x03]
@ CHECK: vst2.32 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x04,0xf9,0xaf,0x03]
@ CHECK: vst2.32 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x04,0xf9,0xbf,0x03]

	vst2.32	{d0, d1, d2, d3}, [r4]!
	vst2.32	{d0, d1, d2, d3}, [r4:16]!
	vst2.32	{d0, d1, d2, d3}, [r4:32]!
	vst2.32	{d0, d1, d2, d3}, [r4:64]!
	vst2.32	{d0, d1, d2, d3}, [r4:128]!
	vst2.32	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vst2.32 {d0, d1, d2, d3}, [r4]! @ encoding: [0x04,0xf9,0x8d,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vst2.32 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x04,0xf9,0x9d,0x03]
@ CHECK: vst2.32 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x04,0xf9,0xad,0x03]
@ CHECK: vst2.32 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x04,0xf9,0xbd,0x03]

	vst2.32	{d0, d1, d2, d3}, [r4], r6
	vst2.32	{d0, d1, d2, d3}, [r4:16], r6
	vst2.32	{d0, d1, d2, d3}, [r4:32], r6
	vst2.32	{d0, d1, d2, d3}, [r4:64], r6
	vst2.32	{d0, d1, d2, d3}, [r4:128], r6
	vst2.32	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vst2.32 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x04,0xf9,0x86,0x03]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst2.32 {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vst2.32 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x04,0xf9,0x96,0x03]
@ CHECK: vst2.32 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x04,0xf9,0xa6,0x03]
@ CHECK: vst2.32 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x04,0xf9,0xb6,0x03]

	vst2.32	{d0[1], d1[1]}, [r4]
	vst2.32	{d0[1], d1[1]}, [r4:16]
	vst2.32	{d0[1], d1[1]}, [r4:32]
	vst2.32	{d0[1], d1[1]}, [r4:64]
	vst2.32	{d0[1], d1[1]}, [r4:128]
	vst2.32	{d0[1], d1[1]}, [r4:256]

@ CHECK: vst2.32 {d0[1], d1[1]}, [r4]    @ encoding: [0x84,0xf9,0x8f,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d1[1]}, [r4:16]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d1[1]}, [r4:32]
@ CHECK-ERRORS:                                     ^
@ CHECK: vst2.32 {d0[1], d1[1]}, [r4:64] @ encoding: [0x84,0xf9,0x9f,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d1[1]}, [r4:128]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d1[1]}, [r4:256]
@ CHECK-ERRORS:                                     ^

	vst2.32	{d0[1], d1[1]}, [r4]!
	vst2.32	{d0[1], d1[1]}, [r4:16]!
	vst2.32	{d0[1], d1[1]}, [r4:32]!
	vst2.32	{d0[1], d1[1]}, [r4:64]!
	vst2.32	{d0[1], d1[1]}, [r4:128]!
	vst2.32	{d0[1], d1[1]}, [r4:256]!

@ CHECK: vst2.32 {d0[1], d1[1]}, [r4]!   @ encoding: [0x84,0xf9,0x8d,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d1[1]}, [r4:16]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d1[1]}, [r4:32]!
@ CHECK-ERRORS:                                     ^
@ CHECK: vst2.32 {d0[1], d1[1]}, [r4:64]! @ encoding: [0x84,0xf9,0x9d,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d1[1]}, [r4:128]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d1[1]}, [r4:256]!
@ CHECK-ERRORS:                                     ^

	vst2.32	{d0[1], d1[1]}, [r4], r6
	vst2.32	{d0[1], d1[1]}, [r4:16], r6
	vst2.32	{d0[1], d1[1]}, [r4:32], r6
	vst2.32	{d0[1], d1[1]}, [r4:64], r6
	vst2.32	{d0[1], d1[1]}, [r4:128], r6
	vst2.32	{d0[1], d1[1]}, [r4:256], r6

@ CHECK: vst2.32 {d0[1], d1[1]}, [r4], r6 @ encoding: [0x84,0xf9,0x86,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d1[1]}, [r4:16], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d1[1]}, [r4:32], r6
@ CHECK-ERRORS:                                     ^
@ CHECK: vst2.32 {d0[1], d1[1]}, [r4:64], r6 @ encoding: [0x84,0xf9,0x96,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d1[1]}, [r4:128], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d1[1]}, [r4:256], r6
@ CHECK-ERRORS:                                     ^

	vst2.32	{d0[1], d2[1]}, [r4]
	vst2.32	{d0[1], d2[1]}, [r4:16]
	vst2.32	{d0[1], d2[1]}, [r4:32]
	vst2.32	{d0[1], d2[1]}, [r4:64]
	vst2.32	{d0[1], d2[1]}, [r4:128]
	vst2.32	{d0[1], d2[1]}, [r4:256]

@ CHECK: vst2.32 {d0[1], d2[1]}, [r4]    @ encoding: [0x84,0xf9,0xcf,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d2[1]}, [r4:16]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d2[1]}, [r4:32]
@ CHECK-ERRORS:                                     ^
@ CHECK: vst2.32 {d0[1], d2[1]}, [r4:64] @ encoding: [0x84,0xf9,0xdf,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d2[1]}, [r4:128]
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d2[1]}, [r4:256]
@ CHECK-ERRORS:                                     ^

	vst2.32	{d0[1], d2[1]}, [r4]!
	vst2.32	{d0[1], d2[1]}, [r4:16]!
	vst2.32	{d0[1], d2[1]}, [r4:32]!
	vst2.32	{d0[1], d2[1]}, [r4:64]!
	vst2.32	{d0[1], d2[1]}, [r4:128]!
	vst2.32	{d0[1], d2[1]}, [r4:256]!

@ CHECK: vst2.32 {d0[1], d2[1]}, [r4]!   @ encoding: [0x84,0xf9,0xcd,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d2[1]}, [r4:16]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d2[1]}, [r4:32]!
@ CHECK-ERRORS:                                     ^
@ CHECK: vst2.32 {d0[1], d2[1]}, [r4:64]! @ encoding: [0x84,0xf9,0xdd,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d2[1]}, [r4:128]!
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d2[1]}, [r4:256]!
@ CHECK-ERRORS:                                     ^

	vst2.32	{d0[1], d2[1]}, [r4], r6
	vst2.32	{d0[1], d2[1]}, [r4:16], r6
	vst2.32	{d0[1], d2[1]}, [r4:32], r6
	vst2.32	{d0[1], d2[1]}, [r4:64], r6
	vst2.32	{d0[1], d2[1]}, [r4:128], r6
	vst2.32	{d0[1], d2[1]}, [r4:256], r6

@ CHECK: vst2.32 {d0[1], d2[1]}, [r4], r6 @ encoding: [0x84,0xf9,0xc6,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d2[1]}, [r4:16], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d2[1]}, [r4:32], r6
@ CHECK-ERRORS:                                     ^
@ CHECK: vst2.32 {d0[1], d2[1]}, [r4:64], r6 @ encoding: [0x84,0xf9,0xd6,0x09]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d2[1]}, [r4:128], r6
@ CHECK-ERRORS:                                     ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst2.32 {d0[1], d2[1]}, [r4:256], r6
@ CHECK-ERRORS:                                     ^

	vst3.8	{d0, d1, d2}, [r4]
	vst3.8	{d0, d1, d2}, [r4:16]
	vst3.8	{d0, d1, d2}, [r4:32]
	vst3.8	{d0, d1, d2}, [r4:64]
	vst3.8	{d0, d1, d2}, [r4:128]
	vst3.8	{d0, d1, d2}, [r4:256]

@ CHECK: vst3.8 {d0, d1, d2}, [r4]      @ encoding: [0x04,0xf9,0x0f,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.8 {d0, d1, d2}, [r4:64]   @ encoding: [0x04,0xf9,0x1f,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vst3.8	{d0, d1, d2}, [r4]!
	vst3.8	{d0, d1, d2}, [r4:16]!
	vst3.8	{d0, d1, d2}, [r4:32]!
	vst3.8	{d0, d1, d2}, [r4:64]!
	vst3.8	{d0, d1, d2}, [r4:128]!
	vst3.8	{d0, d1, d2}, [r4:256]!

@ CHECK: vst3.8 {d0, d1, d2}, [r4]!     @ encoding: [0x04,0xf9,0x0d,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.8 {d0, d1, d2}, [r4:64]!  @ encoding: [0x04,0xf9,0x1d,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vst3.8	{d0, d1, d2}, [r4], r6
	vst3.8	{d0, d1, d2}, [r4:16], r6
	vst3.8	{d0, d1, d2}, [r4:32], r6
	vst3.8	{d0, d1, d2}, [r4:64], r6
	vst3.8	{d0, d1, d2}, [r4:128], r6
	vst3.8	{d0, d1, d2}, [r4:256], r6

@ CHECK: vst3.8 {d0, d1, d2}, [r4], r6  @ encoding: [0x04,0xf9,0x06,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.8 {d0, d1, d2}, [r4:64], r6 @ encoding: [0x04,0xf9,0x16,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vst3.8	{d0, d2, d4}, [r4]
	vst3.8	{d0, d2, d4}, [r4:16]
	vst3.8	{d0, d2, d4}, [r4:32]
	vst3.8	{d0, d2, d4}, [r4:64]
	vst3.8	{d0, d2, d4}, [r4:128]
	vst3.8	{d0, d2, d4}, [r4:256]

@ CHECK: vst3.8 {d0, d2, d4}, [r4]      @ encoding: [0x04,0xf9,0x0f,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d2, d4}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d2, d4}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.8 {d0, d2, d4}, [r4:64]   @ encoding: [0x04,0xf9,0x1f,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d2, d4}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d2, d4}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vst3.8	{d0, d2, d4}, [r4]!
	vst3.8	{d0, d2, d4}, [r4:16]!
	vst3.8	{d0, d2, d4}, [r4:32]!
	vst3.8	{d0, d2, d4}, [r4:64]!
	vst3.8	{d0, d2, d4}, [r4:128]!
	vst3.8	{d0, d2, d4}, [r4:256]!

@ CHECK: vst3.8 {d0, d2, d4}, [r4]!     @ encoding: [0x04,0xf9,0x0d,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d2, d4}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d2, d4}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.8 {d0, d2, d4}, [r4:64]!  @ encoding: [0x04,0xf9,0x1d,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d2, d4}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d2, d4}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vst3.8	{d0, d2, d4}, [r4], r6
	vst3.8	{d0, d2, d4}, [r4:16], r6
	vst3.8	{d0, d2, d4}, [r4:32], r6
	vst3.8	{d0, d2, d4}, [r4:64], r6
	vst3.8	{d0, d2, d4}, [r4:128], r6
	vst3.8	{d0, d2, d4}, [r4:256], r6

@ CHECK: vst3.8 {d0, d2, d4}, [r4], r6  @ encoding: [0x04,0xf9,0x06,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d2, d4}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d2, d4}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.8 {d0, d2, d4}, [r4:64], r6 @ encoding: [0x04,0xf9,0x16,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d2, d4}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.8  {d0, d2, d4}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vst3.8	{d0[1], d1[1], d2[1]}, [r4]
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:16]
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:32]
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:64]
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:128]
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:256]

@ CHECK: vst3.8 {d0[1], d1[1], d2[1]}, [r4] @ encoding: [0x84,0xf9,0x2f,0x02]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:16]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:32]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:64]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:128]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:256]
@ CHECK-ERRORS:                                            ^

	vst3.8	{d0[1], d1[1], d2[1]}, [r4]!
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:16]!
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:32]!
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:64]!
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:128]!
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:256]!

@ CHECK: vst3.8 {d0[1], d1[1], d2[1]}, [r4]! @ encoding: [0x84,0xf9,0x2d,0x02]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:16]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:32]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:64]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:128]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:256]!
@ CHECK-ERRORS:                                            ^

	vst3.8	{d0[1], d1[1], d2[1]}, [r4], r6
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:16], r6
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:32], r6
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:64], r6
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:128], r6
	vst3.8	{d0[1], d1[1], d2[1]}, [r4:256], r6

@ CHECK: vst3.8 {d0[1], d1[1], d2[1]}, [r4], r6 @ encoding: [0x84,0xf9,0x26,0x02]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:16], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:32], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:64], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:128], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.8  {d0[1], d1[1], d2[1]}, [r4:256], r6
@ CHECK-ERRORS:                                            ^

	vst3.16	{d0, d1, d2}, [r4]
	vst3.16	{d0, d1, d2}, [r4:16]
	vst3.16	{d0, d1, d2}, [r4:32]
	vst3.16	{d0, d1, d2}, [r4:64]
	vst3.16	{d0, d1, d2}, [r4:128]
	vst3.16	{d0, d1, d2}, [r4:256]

@ CHECK: vst3.16 {d0, d1, d2}, [r4]      @ encoding: [0x04,0xf9,0x4f,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.16 {d0, d1, d2}, [r4:64]   @ encoding: [0x04,0xf9,0x5f,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vst3.16	{d0, d1, d2}, [r4]!
	vst3.16	{d0, d1, d2}, [r4:16]!
	vst3.16	{d0, d1, d2}, [r4:32]!
	vst3.16	{d0, d1, d2}, [r4:64]!
	vst3.16	{d0, d1, d2}, [r4:128]!
	vst3.16	{d0, d1, d2}, [r4:256]!

@ CHECK: vst3.16 {d0, d1, d2}, [r4]!     @ encoding: [0x04,0xf9,0x4d,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.16 {d0, d1, d2}, [r4:64]!  @ encoding: [0x04,0xf9,0x5d,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vst3.16	{d0, d1, d2}, [r4], r6
	vst3.16	{d0, d1, d2}, [r4:16], r6
	vst3.16	{d0, d1, d2}, [r4:32], r6
	vst3.16	{d0, d1, d2}, [r4:64], r6
	vst3.16	{d0, d1, d2}, [r4:128], r6
	vst3.16	{d0, d1, d2}, [r4:256], r6

@ CHECK: vst3.16 {d0, d1, d2}, [r4], r6  @ encoding: [0x04,0xf9,0x46,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.16 {d0, d1, d2}, [r4:64], r6 @ encoding: [0x04,0xf9,0x56,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vst3.16	{d0, d2, d4}, [r4]
	vst3.16	{d0, d2, d4}, [r4:16]
	vst3.16	{d0, d2, d4}, [r4:32]
	vst3.16	{d0, d2, d4}, [r4:64]
	vst3.16	{d0, d2, d4}, [r4:128]
	vst3.16	{d0, d2, d4}, [r4:256]

@ CHECK: vst3.16 {d0, d2, d4}, [r4]      @ encoding: [0x04,0xf9,0x4f,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d2, d4}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d2, d4}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.16 {d0, d2, d4}, [r4:64]   @ encoding: [0x04,0xf9,0x5f,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d2, d4}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d2, d4}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vst3.16	{d0, d2, d4}, [r4]!
	vst3.16	{d0, d2, d4}, [r4:16]!
	vst3.16	{d0, d2, d4}, [r4:32]!
	vst3.16	{d0, d2, d4}, [r4:64]!
	vst3.16	{d0, d2, d4}, [r4:128]!
	vst3.16	{d0, d2, d4}, [r4:256]!

@ CHECK: vst3.16 {d0, d2, d4}, [r4]!     @ encoding: [0x04,0xf9,0x4d,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d2, d4}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d2, d4}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.16 {d0, d2, d4}, [r4:64]!  @ encoding: [0x04,0xf9,0x5d,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d2, d4}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d2, d4}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vst3.16	{d0, d2, d4}, [r4], r6
	vst3.16	{d0, d2, d4}, [r4:16], r6
	vst3.16	{d0, d2, d4}, [r4:32], r6
	vst3.16	{d0, d2, d4}, [r4:64], r6
	vst3.16	{d0, d2, d4}, [r4:128], r6
	vst3.16	{d0, d2, d4}, [r4:256], r6

@ CHECK: vst3.16 {d0, d2, d4}, [r4], r6  @ encoding: [0x04,0xf9,0x46,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d2, d4}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d2, d4}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.16 {d0, d2, d4}, [r4:64], r6 @ encoding: [0x04,0xf9,0x56,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d2, d4}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.16 {d0, d2, d4}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vst3.16	{d0[1], d1[1], d2[1]}, [r4]
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:16]
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:32]
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:64]
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:128]
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:256]

@ CHECK: vst3.16 {d0[1], d1[1], d2[1]}, [r4] @ encoding: [0x84,0xf9,0x4f,0x06]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:16]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:32]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:64]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:128]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:256]
@ CHECK-ERRORS:                                            ^

	vst3.16	{d0[1], d1[1], d2[1]}, [r4]!
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:16]!
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:32]!
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:64]!
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:128]!
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:256]!

@ CHECK: vst3.16 {d0[1], d1[1], d2[1]}, [r4]! @ encoding: [0x84,0xf9,0x4d,0x06]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:16]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:32]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:64]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:128]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:256]!
@ CHECK-ERRORS:                                            ^

	vst3.16	{d0[1], d1[1], d2[1]}, [r4], r6
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:16], r6
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:32], r6
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:64], r6
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:128], r6
	vst3.16	{d0[1], d1[1], d2[1]}, [r4:256], r6

@ CHECK: vst3.16 {d0[1], d1[1], d2[1]}, [r4], r6 @ encoding: [0x84,0xf9,0x46,0x06]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:16], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:32], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:64], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:128], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d1[1], d2[1]}, [r4:256], r6
@ CHECK-ERRORS:                                            ^

	vst3.16	{d0[1], d2[1], d4[1]}, [r4]
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:16]
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:32]
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:64]
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:128]
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:256]

@ CHECK: vst3.16 {d0[1], d2[1], d4[1]}, [r4] @ encoding: [0x84,0xf9,0x6f,0x06]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:16]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:32]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:64]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:128]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:256]
@ CHECK-ERRORS:                                            ^

	vst3.16	{d0[1], d2[1], d4[1]}, [r4]!
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:16]!
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:32]!
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:64]!
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:128]!
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:256]!

@ CHECK: vst3.16 {d0[1], d1[1], d2[1]}, [r4]! @ encoding: [0x84,0xf9,0x6d,0x06]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:16]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:32]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:64]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:128]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:256]!
@ CHECK-ERRORS:                                            ^

	vst3.16	{d0[1], d2[1], d4[1]}, [r4], r6
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:16], r6
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:32], r6
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:64], r6
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:128], r6
	vst3.16	{d0[1], d2[1], d4[1]}, [r4:256], r6

@ CHECK: vst3.16 {d0[1], d2[1], d4[1]}, [r4], r6 @ encoding: [0x84,0xf9,0x66,0x06]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:16], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:32], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:64], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:128], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.16 {d0[1], d2[1], d4[1]}, [r4:256], r6
@ CHECK-ERRORS:                                            ^

	vst3.32	{d0, d1, d2}, [r4]
	vst3.32	{d0, d1, d2}, [r4:16]
	vst3.32	{d0, d1, d2}, [r4:32]
	vst3.32	{d0, d1, d2}, [r4:64]
	vst3.32	{d0, d1, d2}, [r4:128]
	vst3.32	{d0, d1, d2}, [r4:256]

@ CHECK: vst3.32 {d0, d1, d2}, [r4]      @ encoding: [0x04,0xf9,0x8f,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d1, d2}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d1, d2}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.32 {d0, d1, d2}, [r4:64]   @ encoding: [0x04,0xf9,0x9f,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d1, d2}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d1, d2}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vst3.32	{d0, d1, d2}, [r4]!
	vst3.32	{d0, d1, d2}, [r4:16]!
	vst3.32	{d0, d1, d2}, [r4:32]!
	vst3.32	{d0, d1, d2}, [r4:64]!
	vst3.32	{d0, d1, d2}, [r4:128]!
	vst3.32	{d0, d1, d2}, [r4:256]!

@ CHECK: vst3.32 {d0, d1, d2}, [r4]!     @ encoding: [0x04,0xf9,0x8d,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d1, d2}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d1, d2}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.32 {d0, d1, d2}, [r4:64]!  @ encoding: [0x04,0xf9,0x9d,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d1, d2}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d1, d2}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vst3.32	{d0, d1, d2}, [r4], r6
	vst3.32	{d0, d1, d2}, [r4:16], r6
	vst3.32	{d0, d1, d2}, [r4:32], r6
	vst3.32	{d0, d1, d2}, [r4:64], r6
	vst3.32	{d0, d1, d2}, [r4:128], r6
	vst3.32	{d0, d1, d2}, [r4:256], r6

@ CHECK: vst3.32 {d0, d1, d2}, [r4], r6  @ encoding: [0x04,0xf9,0x86,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d1, d2}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d1, d2}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.32 {d0, d1, d2}, [r4:64], r6 @ encoding: [0x04,0xf9,0x96,0x04]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d1, d2}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d1, d2}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vst3.32	{d0, d2, d4}, [r4]
	vst3.32	{d0, d2, d4}, [r4:16]
	vst3.32	{d0, d2, d4}, [r4:32]
	vst3.32	{d0, d2, d4}, [r4:64]
	vst3.32	{d0, d2, d4}, [r4:128]
	vst3.32	{d0, d2, d4}, [r4:256]

@ CHECK: vst3.32 {d0, d2, d4}, [r4]      @ encoding: [0x04,0xf9,0x8f,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d2, d4}, [r4:16]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d2, d4}, [r4:32]
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.32 {d0, d2, d4}, [r4:64]   @ encoding: [0x04,0xf9,0x9f,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d2, d4}, [r4:128]
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d2, d4}, [r4:256]
@ CHECK-ERRORS:                                   ^

	vst3.32	{d0, d2, d4}, [r4]!
	vst3.32	{d0, d2, d4}, [r4:16]!
	vst3.32	{d0, d2, d4}, [r4:32]!
	vst3.32	{d0, d2, d4}, [r4:64]!
	vst3.32	{d0, d2, d4}, [r4:128]!
	vst3.32	{d0, d2, d4}, [r4:256]!

@ CHECK: vst3.32 {d0, d2, d4}, [r4]!     @ encoding: [0x04,0xf9,0x8d,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d2, d4}, [r4:16]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d2, d4}, [r4:32]!
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.32 {d0, d2, d4}, [r4:64]!  @ encoding: [0x04,0xf9,0x9d,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d2, d4}, [r4:128]!
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d2, d4}, [r4:256]!
@ CHECK-ERRORS:                                   ^

	vst3.32	{d0, d2, d4}, [r4], r6
	vst3.32	{d0, d2, d4}, [r4:16], r6
	vst3.32	{d0, d2, d4}, [r4:32], r6
	vst3.32	{d0, d2, d4}, [r4:64], r6
	vst3.32	{d0, d2, d4}, [r4:128], r6
	vst3.32	{d0, d2, d4}, [r4:256], r6

@ CHECK: vst3.32 {d0, d2, d4}, [r4], r6  @ encoding: [0x04,0xf9,0x86,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d2, d4}, [r4:16], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d2, d4}, [r4:32], r6
@ CHECK-ERRORS:                                   ^
@ CHECK: vst3.32 {d0, d2, d4}, [r4:64], r6 @ encoding: [0x04,0xf9,0x96,0x05]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d2, d4}, [r4:128], r6
@ CHECK-ERRORS:                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst3.32 {d0, d2, d4}, [r4:256], r6
@ CHECK-ERRORS:                                   ^

	vst3.32	{d0[1], d1[1], d2[1]}, [r4]
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:16]
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:32]
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:64]
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:128]
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:256]

@ CHECK: vst3.32 {d0[1], d1[1], d2[1]}, [r4] @ encoding: [0x84,0xf9,0x8f,0x0a]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:16]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:32]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:64]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:128]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:256]
@ CHECK-ERRORS:                                            ^

	vst3.32	{d0[1], d1[1], d2[1]}, [r4]!
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:16]!
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:32]!
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:64]!
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:128]!
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:256]!

@ CHECK: vst3.32 {d0[1], d1[1], d2[1]}, [r4]! @ encoding: [0x84,0xf9,0x8d,0x0a]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:16]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:32]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:64]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:128]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:256]!
@ CHECK-ERRORS:                                            ^

	vst3.32	{d0[1], d1[1], d2[1]}, [r4], r6
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:16], r6
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:32], r6
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:64], r6
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:128], r6
	vst3.32	{d0[1], d1[1], d2[1]}, [r4:256], r6

@ CHECK: vst3.32 {d0[1], d1[1], d2[1]}, [r4], r6 @ encoding: [0x84,0xf9,0x86,0x0a]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:16], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:32], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:64], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:128], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d1[1], d2[1]}, [r4:256], r6
@ CHECK-ERRORS:                                            ^

	vst3.32	{d0[1], d2[1], d4[1]}, [r4]
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:16]
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:32]
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:64]
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:128]
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:256]

@ CHECK: vst3.32 {d0[1], d2[1], d4[1]}, [r4] @ encoding: [0x84,0xf9,0xcf,0x0a]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:16]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:32]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:64]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:128]
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:256]
@ CHECK-ERRORS:                                            ^

	vst3.32	{d0[1], d2[1], d4[1]}, [r4]!
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:16]!
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:32]!
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:64]!
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:128]!
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:256]!

@ CHECK: vst3.32 {d0[1], d2[1], d4[1]}, [r4]! @ encoding: [0x84,0xf9,0xcd,0x0a]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:16]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:32]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:64]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:128]!
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:256]!
@ CHECK-ERRORS:                                            ^

	vst3.32	{d0[1], d2[1], d4[1]}, [r4], r6
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:16], r6
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:32], r6
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:64], r6
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:128], r6
	vst3.32	{d0[1], d2[1], d4[1]}, [r4:256], r6

@ CHECK: vst3.32 {d0[1], d2[1], d4[1]}, [r4], r6 @ encoding: [0x84,0xf9,0xc6,0x0a]
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:16], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:32], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:64], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:128], r6
@ CHECK-ERRORS:                                            ^
@ CHECK-ERRORS: alignment must be omitted
@ CHECK-ERRORS:         vst3.32 {d0[1], d2[1], d4[1]}, [r4:256], r6
@ CHECK-ERRORS:                                            ^

	vst4.8	{d0, d1, d2, d3}, [r4]
	vst4.8	{d0, d1, d2, d3}, [r4:16]
	vst4.8	{d0, d1, d2, d3}, [r4:32]
	vst4.8	{d0, d1, d2, d3}, [r4:64]
	vst4.8	{d0, d1, d2, d3}, [r4:128]
	vst4.8	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vst4.8 {d0, d1, d2, d3}, [r4]  @ encoding: [0x04,0xf9,0x0f,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.8  {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.8  {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.8 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x04,0xf9,0x1f,0x00]
@ CHECK: vst4.8 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x04,0xf9,0x2f,0x00]
@ CHECK: vst4.8 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x04,0xf9,0x3f,0x00]

	vst4.8	{d0, d1, d2, d3}, [r4]!
	vst4.8	{d0, d1, d2, d3}, [r4:16]!
	vst4.8	{d0, d1, d2, d3}, [r4:32]!
	vst4.8	{d0, d1, d2, d3}, [r4:64]!
	vst4.8	{d0, d1, d2, d3}, [r4:128]!
	vst4.8	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vst4.8 {d0, d1, d2, d3}, [r4]! @ encoding: [0x04,0xf9,0x0d,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.8  {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.8  {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.8 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x04,0xf9,0x1d,0x00]
@ CHECK: vst4.8 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x04,0xf9,0x2d,0x00]
@ CHECK: vst4.8 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x04,0xf9,0x3d,0x00]

	vst4.8	{d0, d1, d2, d3}, [r4], r6
	vst4.8	{d0, d1, d2, d3}, [r4:16], r6
	vst4.8	{d0, d1, d2, d3}, [r4:32], r6
	vst4.8	{d0, d1, d2, d3}, [r4:64], r6
	vst4.8	{d0, d1, d2, d3}, [r4:128], r6
	vst4.8	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vst4.8 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x04,0xf9,0x06,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.8  {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.8  {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.8 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x04,0xf9,0x16,0x00]
@ CHECK: vst4.8 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x04,0xf9,0x26,0x00]
@ CHECK: vst4.8 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x04,0xf9,0x36,0x00]

	vst4.8	{d0, d2, d4, d6}, [r4]
	vst4.8	{d0, d2, d4, d6}, [r4:16]
	vst4.8	{d0, d2, d4, d6}, [r4:32]
	vst4.8	{d0, d2, d4, d6}, [r4:64]
	vst4.8	{d0, d2, d4, d6}, [r4:128]
	vst4.8	{d0, d2, d4, d6}, [r4:256]

@ CHECK: vst4.8 {d0, d2, d4, d6}, [r4]  @ encoding: [0x04,0xf9,0x0f,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.8  {d0, d2, d4, d6}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.8  {d0, d2, d4, d6}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.8 {d0, d2, d4, d6}, [r4:64] @ encoding: [0x04,0xf9,0x1f,0x01]
@ CHECK: vst4.8 {d0, d2, d4, d6}, [r4:128] @ encoding: [0x04,0xf9,0x2f,0x01]
@ CHECK: vst4.8 {d0, d2, d4, d6}, [r4:256] @ encoding: [0x04,0xf9,0x3f,0x01]

	vst4.8	{d0, d2, d4, d6}, [r4]!
	vst4.8	{d0, d2, d4, d6}, [r4:16]!
	vst4.8	{d0, d2, d4, d6}, [r4:32]!
	vst4.8	{d0, d2, d4, d6}, [r4:64]!
	vst4.8	{d0, d2, d4, d6}, [r4:128]!
	vst4.8	{d0, d2, d4, d6}, [r4:256]!

@ CHECK: vst4.8 {d0, d2, d4, d6}, [r4]! @ encoding: [0x04,0xf9,0x0d,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.8  {d0, d2, d4, d6}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.8  {d0, d2, d4, d6}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.8 {d0, d2, d4, d6}, [r4:64]! @ encoding: [0x04,0xf9,0x1d,0x01]
@ CHECK: vst4.8 {d0, d2, d4, d6}, [r4:128]! @ encoding: [0x04,0xf9,0x2d,0x01]
@ CHECK: vst4.8 {d0, d2, d4, d6}, [r4:256]! @ encoding: [0x04,0xf9,0x3d,0x01]

	vst4.8	{d0, d2, d4, d6}, [r4], r6
	vst4.8	{d0, d2, d4, d6}, [r4:16], r6
	vst4.8	{d0, d2, d4, d6}, [r4:32], r6
	vst4.8	{d0, d2, d4, d6}, [r4:64], r6
	vst4.8	{d0, d2, d4, d6}, [r4:128], r6
	vst4.8	{d0, d2, d4, d6}, [r4:256], r6

@ CHECK: vst4.8 {d0, d2, d4, d6}, [r4], r6 @ encoding: [0x04,0xf9,0x06,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.8  {d0, d2, d4, d6}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.8  {d0, d2, d4, d6}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.8 {d0, d2, d4, d6}, [r4:64], r6 @ encoding: [0x04,0xf9,0x16,0x01]
@ CHECK: vst4.8 {d0, d2, d4, d6}, [r4:128], r6 @ encoding: [0x04,0xf9,0x26,0x01]
@ CHECK: vst4.8 {d0, d2, d4, d6}, [r4:256], r6 @ encoding: [0x04,0xf9,0x36,0x01]

	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4]
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:16]
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:32]
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:64]
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:128]
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:256]

@ CHECK: vst4.8 {d0[1], d1[1], d2[1], d3[1]}, [r4] @ encoding: [0x84,0xf9,0x2f,0x03]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:16]
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.8 {d0[1], d1[1], d2[1], d3[1]}, [r4:32] @ encoding: [0x84,0xf9,0x3f,0x03]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:64]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:128]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:256]
@ CHECK-ERRORS:                                                   ^

	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4]!
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:32]!
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:64]!
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:128]!
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:256]!

@ CHECK: vst4.8 {d0[1], d1[1], d2[1], d3[1]}, [r4]! @ encoding: [0x84,0xf9,0x2d,0x03]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.8 {d0[1], d1[1], d2[1], d3[1]}, [r4:32]! @ encoding: [0x84,0xf9,0x3d,0x03]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:64]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:128]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:256]!
@ CHECK-ERRORS:                                                   ^

	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4], r6
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6
	vst4.8	{d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6

@ CHECK: vst4.8 {d0[1], d1[1], d2[1], d3[1]}, [r4], r6 @ encoding: [0x84,0xf9,0x26,0x03]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.8 {d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6 @ encoding: [0x84,0xf9,0x36,0x03]
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 32 or omitted
@ CHECK-ERRORS:         vst4.8  {d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6
@ CHECK-ERRORS:                                                   ^

	vst4.16	{d0, d1, d2, d3}, [r4]
	vst4.16	{d0, d1, d2, d3}, [r4:16]
	vst4.16	{d0, d1, d2, d3}, [r4:32]
	vst4.16	{d0, d1, d2, d3}, [r4:64]
	vst4.16	{d0, d1, d2, d3}, [r4:128]
	vst4.16	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vst4.16 {d0, d1, d2, d3}, [r4]  @ encoding: [0x04,0xf9,0x4f,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.16 {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.16 {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.16 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x04,0xf9,0x5f,0x00]
@ CHECK: vst4.16 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x04,0xf9,0x6f,0x00]
@ CHECK: vst4.16 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x04,0xf9,0x7f,0x00]

	vst4.16	{d0, d1, d2, d3}, [r4]!
	vst4.16	{d0, d1, d2, d3}, [r4:16]!
	vst4.16	{d0, d1, d2, d3}, [r4:32]!
	vst4.16	{d0, d1, d2, d3}, [r4:64]!
	vst4.16	{d0, d1, d2, d3}, [r4:128]!
	vst4.16	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vst4.16 {d0, d1, d2, d3}, [r4]! @ encoding: [0x04,0xf9,0x4d,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.16 {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.16 {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.16 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x04,0xf9,0x5d,0x00]
@ CHECK: vst4.16 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x04,0xf9,0x6d,0x00]
@ CHECK: vst4.16 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x04,0xf9,0x7d,0x00]

	vst4.16	{d0, d1, d2, d3}, [r4], r6
	vst4.16	{d0, d1, d2, d3}, [r4:16], r6
	vst4.16	{d0, d1, d2, d3}, [r4:32], r6
	vst4.16	{d0, d1, d2, d3}, [r4:64], r6
	vst4.16	{d0, d1, d2, d3}, [r4:128], r6
	vst4.16	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vst4.16 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x04,0xf9,0x46,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.16 {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.16 {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.16 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x04,0xf9,0x56,0x00]
@ CHECK: vst4.16 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x04,0xf9,0x66,0x00]
@ CHECK: vst4.16 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x04,0xf9,0x76,0x00]

	vst4.16	{d0, d2, d4, d6}, [r4]
	vst4.16	{d0, d2, d4, d6}, [r4:16]
	vst4.16	{d0, d2, d4, d6}, [r4:32]
	vst4.16	{d0, d2, d4, d6}, [r4:64]
	vst4.16	{d0, d2, d4, d6}, [r4:128]
	vst4.16	{d0, d2, d4, d6}, [r4:256]

@ CHECK: vst4.16 {d0, d2, d4, d6}, [r4]  @ encoding: [0x04,0xf9,0x4f,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.16 {d0, d2, d4, d6}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.16 {d0, d2, d4, d6}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.16 {d0, d2, d4, d6}, [r4:64] @ encoding: [0x04,0xf9,0x5f,0x01]
@ CHECK: vst4.16 {d0, d2, d4, d6}, [r4:128] @ encoding: [0x04,0xf9,0x6f,0x01]
@ CHECK: vst4.16 {d0, d2, d4, d6}, [r4:256] @ encoding: [0x04,0xf9,0x7f,0x01]

	vst4.16	{d0, d2, d4, d6}, [r4]!
	vst4.16	{d0, d2, d4, d6}, [r4:16]!
	vst4.16	{d0, d2, d4, d6}, [r4:32]!
	vst4.16	{d0, d2, d4, d6}, [r4:64]!
	vst4.16	{d0, d2, d4, d6}, [r4:128]!
	vst4.16	{d0, d2, d4, d6}, [r4:256]!

@ CHECK: vst4.16 {d0, d2, d4, d6}, [r4]! @ encoding: [0x04,0xf9,0x4d,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.16 {d0, d2, d4, d6}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.16 {d0, d2, d4, d6}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.16 {d0, d2, d4, d6}, [r4:64]! @ encoding: [0x04,0xf9,0x5d,0x01]
@ CHECK: vst4.16 {d0, d2, d4, d6}, [r4:128]! @ encoding: [0x04,0xf9,0x6d,0x01]
@ CHECK: vst4.16 {d0, d2, d4, d6}, [r4:256]! @ encoding: [0x04,0xf9,0x7d,0x01]

	vst4.16	{d0, d2, d4, d6}, [r4], r6
	vst4.16	{d0, d2, d4, d6}, [r4:16], r6
	vst4.16	{d0, d2, d4, d6}, [r4:32], r6
	vst4.16	{d0, d2, d4, d6}, [r4:64], r6
	vst4.16	{d0, d2, d4, d6}, [r4:128], r6
	vst4.16	{d0, d2, d4, d6}, [r4:256], r6

@ CHECK: vst4.16 {d0, d2, d4, d6}, [r4], r6 @ encoding: [0x04,0xf9,0x46,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.16 {d0, d2, d4, d6}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.16 {d0, d2, d4, d6}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.16 {d0, d2, d4, d6}, [r4:64], r6 @ encoding: [0x04,0xf9,0x56,0x01]
@ CHECK: vst4.16 {d0, d2, d4, d6}, [r4:128], r6 @ encoding: [0x04,0xf9,0x66,0x01]
@ CHECK: vst4.16 {d0, d2, d4, d6}, [r4:256], r6 @ encoding: [0x04,0xf9,0x76,0x01]

	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4]
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:16]
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:32]
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:64]
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:128]
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:256]

@ CHECK: vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4] @ encoding: [0x84,0xf9,0x4f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:16]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:32]
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:64] @ encoding: [0x84,0xf9,0x5f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:128]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:256]
@ CHECK-ERRORS:                                                   ^

	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4]!
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:32]!
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:64]!
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:128]!
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:256]!

@ CHECK: vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4]! @ encoding: [0x84,0xf9,0x4d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:32]!
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:64]! @ encoding: [0x84,0xf9,0x5d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:128]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:256]!
@ CHECK-ERRORS:                                                   ^

	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4], r6
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6
	vst4.16	{d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6

@ CHECK: vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4], r6 @ encoding: [0x84,0xf9,0x46,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6 @ encoding: [0x84,0xf9,0x56,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6
@ CHECK-ERRORS:                                                   ^

	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4]
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:16]
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:32]
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:64]
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:128]
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:256]

@ CHECK: vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4] @ encoding: [0x84,0xf9,0x6f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:16]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:32]
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:64] @ encoding: [0x84,0xf9,0x7f,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:128]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:256]
@ CHECK-ERRORS:                                                   ^

	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4]!
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:16]!
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:32]!
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:64]!
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:128]!
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:256]!

@ CHECK: vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4]! @ encoding: [0x84,0xf9,0x6d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:16]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:32]!
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.16 {d0[1], d1[1], d2[1], d3[1]}, [r4:64]! @ encoding: [0x84,0xf9,0x7d,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:128]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:256]!
@ CHECK-ERRORS:                                                   ^

	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4], r6
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:16], r6
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:32], r6
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:64], r6
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:128], r6
	vst4.16	{d0[1], d2[1], d4[1], d6[1]}, [r4:256], r6

@ CHECK: vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4], r6 @ encoding: [0x84,0xf9,0x66,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:16], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:32], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:64], r6 @ encoding: [0x84,0xf9,0x76,0x07]
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:128], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64 or omitted
@ CHECK-ERRORS:         vst4.16 {d0[1], d2[1], d4[1], d6[1]}, [r4:256], r6
@ CHECK-ERRORS:                                                   ^

	vst4.32	{d0, d1, d2, d3}, [r4]
	vst4.32	{d0, d1, d2, d3}, [r4:16]
	vst4.32	{d0, d1, d2, d3}, [r4:32]
	vst4.32	{d0, d1, d2, d3}, [r4:64]
	vst4.32	{d0, d1, d2, d3}, [r4:128]
	vst4.32	{d0, d1, d2, d3}, [r4:256]

@ CHECK: vst4.32 {d0, d1, d2, d3}, [r4]  @ encoding: [0x04,0xf9,0x8f,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.32 {d0, d1, d2, d3}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.32 {d0, d1, d2, d3}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.32 {d0, d1, d2, d3}, [r4:64] @ encoding: [0x04,0xf9,0x9f,0x00]
@ CHECK: vst4.32 {d0, d1, d2, d3}, [r4:128] @ encoding: [0x04,0xf9,0xaf,0x00]
@ CHECK: vst4.32 {d0, d1, d2, d3}, [r4:256] @ encoding: [0x04,0xf9,0xbf,0x00]

	vst4.32	{d0, d1, d2, d3}, [r4]!
	vst4.32	{d0, d1, d2, d3}, [r4:16]!
	vst4.32	{d0, d1, d2, d3}, [r4:32]!
	vst4.32	{d0, d1, d2, d3}, [r4:64]!
	vst4.32	{d0, d1, d2, d3}, [r4:128]!
	vst4.32	{d0, d1, d2, d3}, [r4:256]!

@ CHECK: vst4.32 {d0, d1, d2, d3}, [r4]! @ encoding: [0x04,0xf9,0x8d,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.32 {d0, d1, d2, d3}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.32 {d0, d1, d2, d3}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.32 {d0, d1, d2, d3}, [r4:64]! @ encoding: [0x04,0xf9,0x9d,0x00]
@ CHECK: vst4.32 {d0, d1, d2, d3}, [r4:128]! @ encoding: [0x04,0xf9,0xad,0x00]
@ CHECK: vst4.32 {d0, d1, d2, d3}, [r4:256]! @ encoding: [0x04,0xf9,0xbd,0x00]

	vst4.32	{d0, d1, d2, d3}, [r4], r6
	vst4.32	{d0, d1, d2, d3}, [r4:16], r6
	vst4.32	{d0, d1, d2, d3}, [r4:32], r6
	vst4.32	{d0, d1, d2, d3}, [r4:64], r6
	vst4.32	{d0, d1, d2, d3}, [r4:128], r6
	vst4.32	{d0, d1, d2, d3}, [r4:256], r6

@ CHECK: vst4.32 {d0, d1, d2, d3}, [r4], r6 @ encoding: [0x04,0xf9,0x86,0x00]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.32 {d0, d1, d2, d3}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.32 {d0, d1, d2, d3}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.32 {d0, d1, d2, d3}, [r4:64], r6 @ encoding: [0x04,0xf9,0x96,0x00]
@ CHECK: vst4.32 {d0, d1, d2, d3}, [r4:128], r6 @ encoding: [0x04,0xf9,0xa6,0x00]
@ CHECK: vst4.32 {d0, d1, d2, d3}, [r4:256], r6 @ encoding: [0x04,0xf9,0xb6,0x00]

	vst4.32	{d0, d2, d4, d6}, [r4]
	vst4.32	{d0, d2, d4, d6}, [r4:16]
	vst4.32	{d0, d2, d4, d6}, [r4:32]
	vst4.32	{d0, d2, d4, d6}, [r4:64]
	vst4.32	{d0, d2, d4, d6}, [r4:128]
	vst4.32	{d0, d2, d4, d6}, [r4:256]

@ CHECK: vst4.32 {d0, d2, d4, d6}, [r4]  @ encoding: [0x04,0xf9,0x8f,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.32 {d0, d2, d4, d6}, [r4:16]
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.32 {d0, d2, d4, d6}, [r4:32]
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.32 {d0, d2, d4, d6}, [r4:64] @ encoding: [0x04,0xf9,0x9f,0x01]
@ CHECK: vst4.32 {d0, d2, d4, d6}, [r4:128] @ encoding: [0x04,0xf9,0xaf,0x01]
@ CHECK: vst4.32 {d0, d2, d4, d6}, [r4:256] @ encoding: [0x04,0xf9,0xbf,0x01]

	vst4.32	{d0, d2, d4, d6}, [r4]!
	vst4.32	{d0, d2, d4, d6}, [r4:16]!
	vst4.32	{d0, d2, d4, d6}, [r4:32]!
	vst4.32	{d0, d2, d4, d6}, [r4:64]!
	vst4.32	{d0, d2, d4, d6}, [r4:128]!
	vst4.32	{d0, d2, d4, d6}, [r4:256]!

@ CHECK: vst4.32 {d0, d2, d4, d6}, [r4]! @ encoding: [0x04,0xf9,0x8d,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.32 {d0, d2, d4, d6}, [r4:16]!
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.32 {d0, d2, d4, d6}, [r4:32]!
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.32 {d0, d2, d4, d6}, [r4:64]! @ encoding: [0x04,0xf9,0x9d,0x01]
@ CHECK: vst4.32 {d0, d2, d4, d6}, [r4:128]! @ encoding: [0x04,0xf9,0xad,0x01]
@ CHECK: vst4.32 {d0, d2, d4, d6}, [r4:256]! @ encoding: [0x04,0xf9,0xbd,0x01]

	vst4.32	{d0, d2, d4, d6}, [r4], r6
	vst4.32	{d0, d2, d4, d6}, [r4:16], r6
	vst4.32	{d0, d2, d4, d6}, [r4:32], r6
	vst4.32	{d0, d2, d4, d6}, [r4:64], r6
	vst4.32	{d0, d2, d4, d6}, [r4:128], r6
	vst4.32	{d0, d2, d4, d6}, [r4:256], r6

@ CHECK: vst4.32 {d0, d2, d4, d6}, [r4], r6 @ encoding: [0x04,0xf9,0x86,0x01]
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.32 {d0, d2, d4, d6}, [r4:16], r6
@ CHECK-ERRORS:                                       ^
@ CHECK-ERRORS: alignment must be 64, 128, 256 or omitted
@ CHECK-ERRORS:         vst4.32 {d0, d2, d4, d6}, [r4:32], r6
@ CHECK-ERRORS:                                       ^
@ CHECK: vst4.32 {d0, d2, d4, d6}, [r4:64], r6 @ encoding: [0x04,0xf9,0x96,0x01]
@ CHECK: vst4.32 {d0, d2, d4, d6}, [r4:128], r6 @ encoding: [0x04,0xf9,0xa6,0x01]
@ CHECK: vst4.32 {d0, d2, d4, d6}, [r4:256], r6 @ encoding: [0x04,0xf9,0xb6,0x01]

	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4]
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:16]
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:32]
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:64]
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:128]
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:256]

@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4] @ encoding: [0x84,0xf9,0x8f,0x0b]
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:16]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:32]
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:64] @ encoding: [0x84,0xf9,0x9f,0x0b]
@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:128] @ encoding: [0x84,0xf9,0xaf,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:256]
@ CHECK-ERRORS:                                                   ^

	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4]!
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:32]!
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:64]!
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:128]!
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:256]!

@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4]! @ encoding: [0x84,0xf9,0x8d,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:32]!
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:64]! @ encoding: [0x84,0xf9,0x9d,0x0b]
@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:128]! @ encoding: [0x84,0xf9,0xad,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:256]!
@ CHECK-ERRORS:                                                   ^

	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4], r6
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6

@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4], r6 @ encoding: [0x84,0xf9,0x86,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6 @ encoding: [0x84,0xf9,0x96,0x0b]
@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6 @ encoding: [0x84,0xf9,0xa6,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6
@ CHECK-ERRORS:                                                   ^

	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4]
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:16]
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:32]
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:64]
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:128]
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:256]

@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4] @ encoding: [0x84,0xf9,0xcf,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:16]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:32]
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:64] @ encoding: [0x84,0xf9,0xdf,0x0b]
@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:128] @ encoding: [0x84,0xf9,0xef,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:256]
@ CHECK-ERRORS:                                                   ^

	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4]!
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:16]!
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:32]!
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:64]!
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:128]!
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:256]!

@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4]! @ encoding: [0x84,0xf9,0xcd,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:16]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:32]!
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:64]! @ encoding: [0x84,0xf9,0xdd,0x0b]
@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:128]! @ encoding: [0x84,0xf9,0xed,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:256]!
@ CHECK-ERRORS:                                                   ^

	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4], r6
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:16], r6
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:32], r6
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:64], r6
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:128], r6
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:256], r6

@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4], r6 @ encoding: [0x84,0xf9,0xc6,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:16], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:32], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:64], r6 @ encoding: [0x84,0xf9,0xd6,0x0b]
@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:128], r6 @ encoding: [0x84,0xf9,0xe6,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:256], r6
@ CHECK-ERRORS:                                                   ^

	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4]!
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:32]!
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:64]!
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:128]!
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:256]!

@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4]! @ encoding: [0x84,0xf9,0x8d,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:16]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:32]!
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:64]! @ encoding: [0x84,0xf9,0x9d,0x0b]
@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:128]! @ encoding: [0x84,0xf9,0xad,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:256]!
@ CHECK-ERRORS:                                                   ^

	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4], r6
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6
	vst4.32	{d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6

@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4], r6 @ encoding: [0x84,0xf9,0x86,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:16], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:32], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:64], r6 @ encoding: [0x84,0xf9,0x96,0x0b]
@ CHECK: vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:128], r6 @ encoding: [0x84,0xf9,0xa6,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d1[1], d2[1], d3[1]}, [r4:256], r6
@ CHECK-ERRORS:                                                   ^

	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4]
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:16]
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:32]
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:64]
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:128]
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:256]

@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4] @ encoding: [0x84,0xf9,0xcf,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:16]
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:32]
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:64] @ encoding: [0x84,0xf9,0xdf,0x0b]
@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:128] @ encoding: [0x84,0xf9,0xef,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:256]
@ CHECK-ERRORS:                                                   ^

	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4]!
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:16]!
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:32]!
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:64]!
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:128]!
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:256]!

@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4]! @ encoding: [0x84,0xf9,0xcd,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:16]!
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:32]!
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:64]! @ encoding: [0x84,0xf9,0xdd,0x0b]
@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:128]! @ encoding: [0x84,0xf9,0xed,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:256]!
@ CHECK-ERRORS:                                                   ^

	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4], r6
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:16], r6
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:32], r6
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:64], r6
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:128], r6
	vst4.32	{d0[1], d2[1], d4[1], d6[1]}, [r4:256], r6

@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4], r6 @ encoding: [0x84,0xf9,0xc6,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:16], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:32], r6
@ CHECK-ERRORS:                                                   ^
@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:64], r6 @ encoding: [0x84,0xf9,0xd6,0x0b]
@ CHECK: vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:128], r6 @ encoding: [0x84,0xf9,0xe6,0x0b]
@ CHECK-ERRORS: alignment must be 64, 128 or omitted
@ CHECK-ERRORS:         vst4.32 {d0[1], d2[1], d4[1], d6[1]}, [r4:256], r6
@ CHECK-ERRORS:                                                   ^
