; RUN: llc < %s -mtriple=i386-pc-mingw32      | FileCheck %s -check-prefix=COFF
; RUN: llc < %s -mtriple=i386-pc-mingw32-elf  | FileCheck %s -check-prefix=ELF

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"

define void @foo1(i32 %N) nounwind {
entry:
; COFF: _foo1:
; COFF: calll __alloca
; ELF: foo1:
; ELF: calll _alloca
	%tmp14 = alloca i32, i32 %N		; <ptr> [#uses=1]
	call void @bar1( ptr %tmp14 )
	ret void
}

declare void @bar1(ptr)

define void @foo2(i32 inreg  %N) nounwind {
entry:
; COFF: _foo2:
; COFF: andl $-16, %esp
; COFF: pushl %eax
; COFF: calll __alloca
; COFF: movl	8012(%esp), %eax
; ELF: foo2:
; ELF: andl $-16, %esp
; ELF: pushl %eax
; ELF: calll _alloca
; ELF: movl	8012(%esp), %eax
	%A2 = alloca [2000 x i32], align 16		; <ptr> [#uses=1]
	%A2.sub = getelementptr [2000 x i32], ptr %A2, i32 0, i32 0		; <ptr> [#uses=1]
	call void @bar2( ptr %A2.sub, i32 %N )
	ret void
}

declare void @bar2(ptr, i32)
