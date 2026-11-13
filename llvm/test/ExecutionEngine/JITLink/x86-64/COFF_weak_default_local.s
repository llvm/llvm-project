# Check that the ".weak.<name>.default[.<suffix>]" helpers MC emits for weak
# definitions do not clash across objects, and that a real weak alias default
# (wbar_real) stays exported. d.o and e.o share the COMDAT suffix "inl".
#
# REQUIRES: asserts
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t/b.s -o %t/b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t/c.s -o %t/c.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t/d.s -o %t/d.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t/e.s -o %t/e.o
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:              -check=%t/main.s %t/main.o %t/a.o %t/b.o %t/c.o \
# RUN:              %t/d.o %t/e.o 2>&1 | FileCheck %s
#
# CHECK-DAG: scope: local, {{.*}} .weak.wfunc.default{{$}}
# CHECK-DAG: scope: local, {{.*}} .weak.wcomdat.default.inl{{$}}
# CHECK-DAG: scope: default, {{.*}} wbar_real{{$}}

#--- main.s
	.text
	.globl	main
	.p2align	4, 0x90
main:
	callq	wfunc
	callq	wbar_real
	callq	wcomdat
	retq

# jitlink-check: *{4}(wfunc + 1) = 1
# jitlink-check: *{4}(wcomdat + 1) = 3

#--- a.s
	.text
	.def	wfunc;
	.scl	2;
	.type	32;
	.endef
	.weak	wfunc
wfunc:
	movl	$1, %eax
	retq

#--- b.s
	.text
	.def	wfunc;
	.scl	2;
	.type	32;
	.endef
	.weak	wfunc
wfunc:
	movl	$2, %eax
	retq

#--- c.s
	.text
	.def	wbar_real;
	.scl	2;
	.type	32;
	.endef
	.globl	wbar_real
wbar_real:
	retq

	.weak	walias
walias = wbar_real

#--- d.s
	.text
	.def	wcomdat;
	.scl	2;
	.type	32;
	.endef
	.weak	wcomdat
wcomdat:
	movl	$3, %eax
	retq

	.section	.text,"xr",discard,inl
	.globl	inl
inl:
	retq

#--- e.s
	.text
	.def	wcomdat;
	.scl	2;
	.type	32;
	.endef
	.weak	wcomdat
wcomdat:
	movl	$4, %eax
	retq

	.section	.text,"xr",discard,inl
	.globl	inl
inl:
	retq
