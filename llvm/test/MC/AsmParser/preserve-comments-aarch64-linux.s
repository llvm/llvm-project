	// REQUIRES: aarch64-registered-target
	// llvm-mc prints leading tabs, formatting of this test follows preserve-comments.s
	// RUN: llvm-mc -preserve-comments -n -triple aarch64-unknown-linux-gnu %s -o %t
	// RUN: diff -b %s %t

	.text

foo:
	// comment here
	nop
	// comment here too
