# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=arm64e-apple-darwin -filetype=obj -o %t/main.o %s
# RUN: llvm-mc -triple=arm64-apple-darwin -filetype=obj -o %t/x.arm64.o \
# RUN:     %S/Inputs/x-1.s
# RUN: llvm-ar crs %t/libX.arm64.a %t/x.arm64.o
# RUN: llvm-mc -triple=arm64e-apple-darwin -filetype=obj -o %t/x.arm64e.o \
# RUN:     %S/Inputs/x-0.s
# RUN: llvm-ar crs %t/libX.arm64e.a %t/x.arm64e.o
# RUN: llvm-lipo --create --output %t/libX.a %t/libX.arm64.a %t/libX.arm64e.a
# RUN: llvm-jitlink -noexec -check=%s %t/main.o -L%t -lX
#
# Create a universal archive with two slices (arm64e, arm64) each containing
# a definition of X: in arm64e X = 0, in arm64 X = 1.
# Check that if we load an arm64e object file then we link the arm64e slice
# of the archive by verifying that X = 0.
#

# jitlink-check: *{4}x = 0

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	2
_main:
	mov     w0, #0
        ret

	.section	__DATA,__data
	.globl	p
p:
	.quad   x

.subsections_via_symbols
