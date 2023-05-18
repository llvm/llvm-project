# RUN: llvm-mc %s -filetype=obj -o - -triple i686-windows-msvc | llvm-nm - | FileCheck %s
# RUN: llvm-mc %s -filetype=obj -o - -triple i686-windows-gnu | llvm-nm - | FileCheck %s

# CHECK: T _mystdcall@8{{$}}
# CHECK: T foo

.text
.global _mystdcall@8
_mystdcall@8:
	movl 4(%esp), %eax
	addl 8(%esp), %eax
	retl $8

.global foo
foo:
	pushl $1
	pushl $2
	calll _mystdcall@8
	retl
