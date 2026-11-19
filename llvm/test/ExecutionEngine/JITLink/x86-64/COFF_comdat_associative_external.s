# An external symbol in a section associative to a COMDAT is part of that
# COMDAT group and takes the linkage of the group's root leader.
#
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t/b.s -o %t/b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t/nodup.s \
# RUN:   -o %t/nodup.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t/nodup-copy.s \
# RUN:   -o %t/nodup-copy.o
#
# With an ANY leader f is weak: b.o's call binds to a.o's f. g, associative to
# f's section, is weak too. b.o's associative section precedes its leader.
#
# RUN: llvm-jitlink -noexec -slab-allocate 100Kb -slab-address 0xfff00000 \
# RUN:   -slab-page-size 4096 -check %s %t/a.o %t/b.o
# RUN: %if asserts %{ llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:   %t/a.o %t/b.o 2>&1 | FileCheck %s --check-prefix=ANY %}
#
# jitlink-check: decode_operand(main, 0) = other - next_pc(main)
# jitlink-check: decode_operand(other, 0) = f - next_pc(other)
# jitlink-check: *{4}(f + 1) = 1
#
# ANY:      Creating defined graph symbol for COFF symbol "g" in .text$g
# ANY-NEXT:   linkage: weak, scope: default, dead - g
# ANY:      Creating defined graph symbol for COFF symbol "f" in .text$f
# ANY-NEXT:   linkage: weak, scope: default, dead - f
# ANY:      Creating defined graph symbol for COFF symbol "f" in .text$f
# ANY-NEXT:   linkage: weak, scope: default, dead - f
#
# With a NODUPLICATES leader f is strong. Two groups (leaders x and y) that
# both define f are a duplicate definition in ORC's interface.
#
# RUN: llvm-jitlink -noexec -entry=other %t/nodup.o
# RUN: %if asserts %{ llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:   -entry=other %t/nodup.o 2>&1 | FileCheck %s --check-prefix=NODUP %}
# RUN: not llvm-jitlink -noexec -entry=other %t/nodup.o %t/nodup-copy.o 2>&1 \
# RUN:   | FileCheck %s --check-prefix=DUP
#
# NODUP:      Creating defined graph symbol for COFF symbol "f" in .text$f
# NODUP-NEXT:   linkage: strong, scope: default, dead - f
#
# DUP: duplicate definition of symbol 'f'
#
# A section associative to a non-COMDAT section is always kept with it, so f
# is strong.
#
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t/noncomdat.s \
# RUN:   -o %t/noncomdat.o
# RUN: llvm-jitlink -noexec %t/noncomdat.o
# RUN: %if asserts %{ llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:   %t/noncomdat.o 2>&1 | FileCheck %s --check-prefix=NONCOMDAT %}
#
# NONCOMDAT:      Creating defined graph symbol for COFF symbol "f" in .text$f
# NONCOMDAT-NEXT:   linkage: strong, scope: default, dead - f

#--- a.s
	.text
	.def main;
	.scl 2;
	.type 32;
	.endef
	.globl main
main:
	callq other
	callq g
	retq

	.section .bss,"bw",discard,x
	.globl x
	.p2align 2
x:
	.long 0

	.section .text$f,"xr",associative,x
	.def f;
	.scl 2;
	.type 32;
	.endef
	.globl f
f:
	movl $1, %eax
	retq

	.section .text$g,"xr",associative,f
	.def g;
	.scl 2;
	.type 32;
	.endef
	.globl g
g:
	retq

#--- b.s
	.section .text$f,"xr",associative,x
	.def f;
	.scl 2;
	.type 32;
	.endef
	.globl f
f:
	movl $2, %eax
	retq

	.section .bss,"bw",discard,x
	.globl x
	.p2align 2
x:
	.long 0

	.text
	.def other;
	.scl 2;
	.type 32;
	.endef
	.globl other
other:
	callq f
	retq

#--- nodup.s
	.text
	.def other;
	.scl 2;
	.type 32;
	.endef
	.globl other
other:
	callq f
	retq

	.section .bss,"bw",one_only,x
	.globl x
	.p2align 2
x:
	.long 0

	.section .text$f,"xr",associative,x
	.def f;
	.scl 2;
	.type 32;
	.endef
	.globl f
f:
	retq

#--- nodup-copy.s
	.section .bss,"bw",one_only,y
	.globl y
	.p2align 2
y:
	.long 0

	.section .text$f,"xr",associative,y
	.def f;
	.scl 2;
	.type 32;
	.endef
	.globl f
f:
	retq

#--- noncomdat.s
	.text
	.def main;
	.scl 2;
	.type 32;
	.endef
	.globl main
main:
	callq f
	retq

	.data
	.globl dvar
dvar:
	.long 0

	.section .text$f,"xr",associative,dvar
	.def f;
	.scl 2;
	.type 32;
	.endef
	.globl f
f:
	retq
