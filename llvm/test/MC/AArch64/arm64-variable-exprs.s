// RUN: llvm-mc -triple arm64-apple-darwin10 %s -filetype=obj -o %t.o

.data

        .long 0
a:
        .long 0
b = a

c:      .long b

d2 = d
.globl d2
d3 = d + 4
.globl d3

e = a + 4

g:
f = g
        .long 0

        .long b
        .long e
        .long a + 4
        .long d
        .long d2
        .long d3
        .long f
        .long g

///
        .text
t0:
Lt0_a:
        .long 0

	.section	__DWARF,__debug_frame,regular,debug
Lt1 = Lt0_a
	.long	Lt1
