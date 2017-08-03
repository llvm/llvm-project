; RUN: llc < %s -asm-verbose=0 -mcpu=generic -mtriple=i686-unknown-linux-gnu -march=x86 -relocation-model=static -code-model=small -pre-RA-sched=list-ilp | FileCheck %s -check-prefix=LINUX-32-STATIC
; RUN: llc < %s -asm-verbose=0 -mcpu=generic -mtriple=i686-unknown-linux-gnu -march=x86 -relocation-model=static -code-model=small -pre-RA-sched=list-ilp | FileCheck %s -check-prefix=LINUX-32-PIC

; RUN: llc < %s -asm-verbose=0 -mcpu=generic -mtriple=x86_64-unknown-linux-gnu -march=x86-64 -relocation-model=static -code-model=small -pre-RA-sched=list-ilp | FileCheck %s -check-prefix=LINUX-64-STATIC
; RUN: llc < %s -asm-verbose=0 -mcpu=generic -mtriple=x86_64-unknown-linux-gnu -march=x86-64 -relocation-model=pic -code-model=small -pre-RA-sched=list-ilp | FileCheck %s -check-prefix=LINUX-64-PIC

; RUN: llc < %s -asm-verbose=0 -mcpu=generic -mtriple=i686-apple-darwin -march=x86 -relocation-model=static -code-model=small -pre-RA-sched=list-ilp | FileCheck %s -check-prefix=DARWIN-32-STATIC
; RUN: llc < %s -asm-verbose=0 -mcpu=generic -mtriple=i686-apple-darwin9 -march=x86 -relocation-model=dynamic-no-pic -code-model=small -pre-RA-sched=list-ilp | FileCheck %s -check-prefix=DARWIN-32-DYNAMIC
; RUN: llc < %s -asm-verbose=0 -mcpu=generic -mtriple=i686-apple-darwin9 -march=x86 -relocation-model=pic -code-model=small -pre-RA-sched=list-ilp | FileCheck %s -check-prefix=DARWIN-32-PIC

; RUN: llc < %s -asm-verbose=0 -mcpu=generic -mtriple=x86_64-apple-darwin -march=x86-64 -relocation-model=static -code-model=small -pre-RA-sched=list-ilp | FileCheck %s -check-prefix=DARWIN-64-STATIC
; RUN: llc < %s -asm-verbose=0 -mcpu=generic -mtriple=x86_64-apple-darwin -march=x86-64 -relocation-model=dynamic-no-pic -code-model=small -pre-RA-sched=list-ilp | FileCheck %s -check-prefix=DARWIN-64-DYNAMIC
; RUN: llc < %s -asm-verbose=0 -mcpu=generic -mtriple=x86_64-apple-darwin -march=x86-64 -relocation-model=pic -code-model=small -pre-RA-sched=list-ilp | FileCheck %s -check-prefix=DARWIN-64-PIC

@src = external global [131072 x i32]
@dst = external global [131072 x i32]
@xsrc = external global [32 x i32]
@xdst = external global [32 x i32]
@ptr = external global i32*
@dsrc = global [131072 x i32] zeroinitializer, align 32
@ddst = global [131072 x i32] zeroinitializer, align 32
@dptr = global i32* null
@lsrc = internal global [131072 x i32] zeroinitializer
@ldst = internal global [131072 x i32] zeroinitializer
@lptr = internal global i32* null
@ifunc = external global void ()*
@difunc = global void ()* null
@lifunc = internal global void ()* null
@lxsrc = internal global [32 x i32] zeroinitializer, align 32
@lxdst = internal global [32 x i32] zeroinitializer, align 32
@dxsrc = global [32 x i32] zeroinitializer, align 32
@dxdst = global [32 x i32] zeroinitializer, align 32

define void @foo00() nounwind {
entry:
	%0 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @src, i32 0, i64 0), align 4
	store i32 %0, i32* getelementptr ([131072 x i32], [131072 x i32]* @dst, i32 0, i64 0), align 4
	ret void

; LINUX-64-STATIC-LABEL: foo00:
; LINUX-64-STATIC: movl	src(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movl	[[EAX]], dst
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: foo00:
; LINUX-32-STATIC: 	movl	src, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], dst
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: foo00:
; LINUX-32-PIC: 	movl	src, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], dst
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: foo00:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r..]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]]), [[EAX:%e..]]
; LINUX-64-PIC-NEXT: 	movq	dst@GOTPCREL(%rip), [[RCX:%r..]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _foo00:
; DARWIN-32-STATIC: 	movl	_src, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _dst
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _foo00:
; DARWIN-32-DYNAMIC: 	movl	L_src$non_lazy_ptr, [[EAX:%e..]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[EAX]]), [[EAX:%e..]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_dst$non_lazy_ptr, [[ECX:%e..]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _foo00:
; DARWIN-32-PIC: 	calll	L0$pb
; DARWIN-32-PIC-NEXT: L0$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e..]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L0$pb([[EAX]]), [[ECX:%e..]]
; DARWIN-32-PIC-NEXT: 	movl	([[ECX]]), [[ECX:%e..]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L0$pb([[EAX]]), [[EAX:%e..]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], ([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _foo00:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r..]]
; DARWIN-64-STATIC-NEXT: 	movl	([[RAX]]), [[EAX:%e..]]
; DARWIN-64-STATIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r..]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _foo00:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r..]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	([[RAX]]), [[EAX:%e..]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r..]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _foo00:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r..]]
; DARWIN-64-PIC-NEXT: 	movl	([[RAX]]), [[EAX:%e..]]
; DARWIN-64-PIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r..]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @fxo00() nounwind {
entry:
	%0 = load i32, i32* getelementptr ([32 x i32], [32 x i32]* @xsrc, i32 0, i64 0), align 4
	store i32 %0, i32* getelementptr ([32 x i32], [32 x i32]* @xdst, i32 0, i64 0), align 4
	ret void

; LINUX-64-STATIC-LABEL: fxo00:
; LINUX-64-STATIC: movl	xsrc(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movl	[[EAX]], xdst
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: fxo00:
; LINUX-32-STATIC: 	movl	xsrc, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], xdst
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: fxo00:
; LINUX-32-PIC: 	movl	xsrc, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], xdst
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: fxo00:
; LINUX-64-PIC: 	movq	xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _fxo00:
; DARWIN-32-STATIC: 	movl	_xsrc, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _xdst
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _fxo00:
; DARWIN-32-DYNAMIC: 	movl	L_xsrc$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_xdst$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _fxo00:
; DARWIN-32-PIC: 	calll	L1$pb
; DARWIN-32-PIC-NEXT: L1$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xsrc$non_lazy_ptr-L1$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[ECX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xdst$non_lazy_ptr-L1$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], ([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _fxo00:
; DARWIN-64-STATIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _fxo00:
; DARWIN-64-DYNAMIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _fxo00:
; DARWIN-64-PIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @foo01() nounwind {
entry:
	store i32* getelementptr ([131072 x i32], [131072 x i32]* @dst, i32 0, i32 0), i32** @ptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: foo01:
; LINUX-64-STATIC: movq	$dst, ptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: foo01:
; LINUX-32-STATIC: 	movl	$dst, ptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: foo01:
; LINUX-32-PIC: 	movl	$dst, ptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: foo01:
; LINUX-64-PIC: 	movq	dst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _foo01:
; DARWIN-32-STATIC: 	movl	$_dst, _ptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _foo01:
; DARWIN-32-DYNAMIC: 	movl	L_dst$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _foo01:
; DARWIN-32-PIC: 	calll	L2$pb
; DARWIN-32-PIC-NEXT: L2$pb:
; DARWIN-32-PIC-NEXT: 	popl
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L2$pb(
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L2$pb(
; DARWIN-32-PIC-NEXT: 	movl
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _foo01:
; DARWIN-64-STATIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _foo01:
; DARWIN-64-DYNAMIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _foo01:
; DARWIN-64-PIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @fxo01() nounwind {
entry:
	store i32* getelementptr ([32 x i32], [32 x i32]* @xdst, i32 0, i32 0), i32** @ptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: fxo01:
; LINUX-64-STATIC: movq	$xdst, ptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: fxo01:
; LINUX-32-STATIC: 	movl	$xdst, ptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: fxo01:
; LINUX-32-PIC: 	movl	$xdst, ptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: fxo01:
; LINUX-64-PIC: 	movq	xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _fxo01:
; DARWIN-32-STATIC: 	movl	$_xdst, _ptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _fxo01:
; DARWIN-32-DYNAMIC: 	movl	L_xdst$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _fxo01:
; DARWIN-32-PIC: 	calll	L3$pb
; DARWIN-32-PIC-NEXT: L3$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[R0:%e..]]
; DARWIN-32-PIC-NEXT: 	movl	L_xdst$non_lazy_ptr-L3$pb([[R0]]), [[R1:%e..]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L3$pb([[R0]]), [[R2:%e..]]
; DARWIN-32-PIC-NEXT: 	movl	[[R1:%e..]], ([[R2]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _fxo01:
; DARWIN-64-STATIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _fxo01:
; DARWIN-64-DYNAMIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _fxo01:
; DARWIN-64-PIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @foo02() nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @src, i32 0, i64 0), align 4
	store i32 %1, i32* %0, align 4
	ret void
; LINUX-64-STATIC-LABEL: foo02:
; LINUX-64-STATIC: movl    src(%rip), %
; LINUX-64-STATIC: movq    ptr(%rip), %
; LINUX-64-STATIC: movl
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: foo02:
; LINUX-32-STATIC: 	movl	src, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: foo02:
; LINUX-32-PIC: 	movl	src, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: foo02:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _foo02:
; DARWIN-32-STATIC: 	movl	_src, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_ptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _foo02:
; DARWIN-32-DYNAMIC: 	movl	L_src$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[ECX]]), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _foo02:
; DARWIN-32-PIC: 	calll	L4$pb
; DARWIN-32-PIC-NEXT: L4$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[R0:%e..]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L4$pb([[R0]]), [[R1:%e..]]
; DARWIN-32-PIC-NEXT: 	movl	([[R1]]), [[R2:%e..]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L4$pb([[R0]]), [[R3:%e..]]
; DARWIN-32-PIC-NEXT: 	movl	([[R3]]), [[R4:%e..]]
; DARWIN-32-PIC-NEXT: 	movl	[[R2]], ([[R4]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _foo02:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _foo02:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _foo02:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @fxo02() nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = load i32, i32* getelementptr ([32 x i32], [32 x i32]* @xsrc, i32 0, i64 0), align 4
	store i32 %1, i32* %0, align 4
; LINUX-64-STATIC-LABEL: fxo02:
; LINUX-64-STATIC: movl    xsrc(%rip), %
; LINUX-64-STATIC: movq    ptr(%rip), %
; LINUX-64-STATIC: movl
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: fxo02:
; LINUX-32-STATIC: 	movl	xsrc, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; LINUX-32-STATIC-NEXT: 	ret
	ret void

; LINUX-32-PIC-LABEL: fxo02:
; LINUX-32-PIC: 	movl	xsrc, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: fxo02:
; LINUX-64-PIC: 	movq	xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _fxo02:
; DARWIN-32-STATIC: 	movl	_xsrc, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_ptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _fxo02:
; DARWIN-32-DYNAMIC: 	movl	L_xsrc$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[ECX]]), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _fxo02:
; DARWIN-32-PIC: 	calll	L5$pb
; DARWIN-32-PIC-NEXT: L5$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xsrc$non_lazy_ptr-L5$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[ECX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L5$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], ([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _fxo02:
; DARWIN-64-STATIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _fxo02:
; DARWIN-64-DYNAMIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _fxo02:
; DARWIN-64-PIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @foo03() nounwind {
entry:
	%0 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @dsrc, i32 0, i64 0), align 32
	store i32 %0, i32* getelementptr ([131072 x i32], [131072 x i32]* @ddst, i32 0, i64 0), align 32
	ret void
; LINUX-64-STATIC-LABEL: foo03:
; LINUX-64-STATIC: movl    dsrc(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], ddst
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: foo03:
; LINUX-32-STATIC: 	movl	dsrc, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ddst
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: foo03:
; LINUX-32-PIC: 	movl	dsrc, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ddst
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: foo03:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ddst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _foo03:
; DARWIN-32-STATIC: 	movl	_dsrc, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _ddst
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _foo03:
; DARWIN-32-DYNAMIC: 	movl	_dsrc, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], _ddst
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _foo03:
; DARWIN-32-PIC: 	calll	L6$pb
; DARWIN-32-PIC-NEXT: L6$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dsrc-L6$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _ddst-L6$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _foo03:
; DARWIN-64-STATIC: 	movl	_dsrc(%rip), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], _ddst(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _foo03:
; DARWIN-64-DYNAMIC: 	movl	_dsrc(%rip), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], _ddst(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _foo03:
; DARWIN-64-PIC: 	movl	_dsrc(%rip), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], _ddst(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @foo04() nounwind {
entry:
	store i32* getelementptr ([131072 x i32], [131072 x i32]* @ddst, i32 0, i32 0), i32** @dptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: foo04:
; LINUX-64-STATIC: movq    $ddst, dptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: foo04:
; LINUX-32-STATIC: 	movl	$ddst, dptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: foo04:
; LINUX-32-PIC: 	movl	$ddst, dptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: foo04:
; LINUX-64-PIC: 	movq	ddst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	dptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _foo04:
; DARWIN-32-STATIC: 	movl	$_ddst, _dptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _foo04:
; DARWIN-32-DYNAMIC: 	movl	$_ddst, _dptr
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _foo04:
; DARWIN-32-PIC: 	calll	L7$pb
; DARWIN-32-PIC-NEXT: L7$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_ddst-L7$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _dptr-L7$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _foo04:
; DARWIN-64-STATIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _foo04:
; DARWIN-64-DYNAMIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _foo04:
; DARWIN-64-PIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @foo05() nounwind {
entry:
	%0 = load i32*, i32** @dptr, align 8
	%1 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @dsrc, i32 0, i64 0), align 32
	store i32 %1, i32* %0, align 4
	ret void
; LINUX-64-STATIC-LABEL: foo05:
; LINUX-64-STATIC: movl    dsrc(%rip), %
; LINUX-64-STATIC: movq    dptr(%rip), %
; LINUX-64-STATIC: movl
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: foo05:
; LINUX-32-STATIC: 	movl	dsrc, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: foo05:
; LINUX-32-PIC: 	movl	dsrc, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: foo05:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	dptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _foo05:
; DARWIN-32-STATIC: 	movl	_dsrc, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _foo05:
; DARWIN-32-DYNAMIC: 	movl	_dsrc, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _foo05:
; DARWIN-32-PIC: 	calll	L8$pb
; DARWIN-32-PIC-NEXT: L8$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dsrc-L8$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dptr-L8$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], ([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _foo05:
; DARWIN-64-STATIC: 	movl	_dsrc(%rip), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _foo05:
; DARWIN-64-DYNAMIC: 	movl	_dsrc(%rip), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _foo05:
; DARWIN-64-PIC: 	movl	_dsrc(%rip), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @foo06() nounwind {
entry:
	%0 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @lsrc, i32 0, i64 0), align 4
	store i32 %0, i32* getelementptr ([131072 x i32], [131072 x i32]* @ldst, i32 0, i64 0), align 4
	ret void
; LINUX-64-STATIC-LABEL: foo06:
; LINUX-64-STATIC: movl    lsrc(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], ldst(%rip)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: foo06:
; LINUX-32-STATIC: 	movl	lsrc, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ldst
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: foo06:
; LINUX-32-PIC: 	movl	lsrc, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ldst
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: foo06:
; LINUX-64-PIC: 	movl	lsrc(%rip), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ldst(%rip)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _foo06:
; DARWIN-32-STATIC: 	movl	_lsrc, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _ldst
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _foo06:
; DARWIN-32-DYNAMIC: 	movl	_lsrc, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], _ldst
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _foo06:
; DARWIN-32-PIC: 	calll	L9$pb
; DARWIN-32-PIC-NEXT: L9$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lsrc-L9$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _ldst-L9$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _foo06:
; DARWIN-64-STATIC: 	movl	_lsrc(%rip), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], _ldst(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _foo06:
; DARWIN-64-DYNAMIC: 	movl	_lsrc(%rip), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], _ldst(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _foo06:
; DARWIN-64-PIC: 	movl	_lsrc(%rip), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], _ldst(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @foo07() nounwind {
entry:
	store i32* getelementptr ([131072 x i32], [131072 x i32]* @ldst, i32 0, i32 0), i32** @lptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: foo07:
; LINUX-64-STATIC: movq    $ldst, lptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: foo07:
; LINUX-32-STATIC: 	movl	$ldst, lptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: foo07:
; LINUX-32-PIC: 	movl	$ldst, lptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: foo07:
; LINUX-64-PIC: 	leaq	ldst(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], lptr(%rip)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _foo07:
; DARWIN-32-STATIC: 	movl	$_ldst, _lptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _foo07:
; DARWIN-32-DYNAMIC: 	movl	$_ldst, _lptr
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _foo07:
; DARWIN-32-PIC: 	calll	L10$pb
; DARWIN-32-PIC-NEXT: L10$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_ldst-L10$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _lptr-L10$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _foo07:
; DARWIN-64-STATIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _foo07:
; DARWIN-64-DYNAMIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _foo07:
; DARWIN-64-PIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @foo08() nounwind {
entry:
	%0 = load i32*, i32** @lptr, align 8
	%1 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @lsrc, i32 0, i64 0), align 4
	store i32 %1, i32* %0, align 4
	ret void
; LINUX-64-STATIC-LABEL: foo08:
; LINUX-64-STATIC: movl    lsrc(%rip), %
; LINUX-64-STATIC: movq    lptr(%rip), %
; LINUX-64-STATIC: movl
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: foo08:
; LINUX-32-STATIC: 	movl	lsrc, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: foo08:
; LINUX-32-PIC: 	movl	lsrc, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: foo08:
; LINUX-64-PIC: 	movl	lsrc(%rip), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	lptr(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _foo08:
; DARWIN-32-STATIC: 	movl	_lsrc, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _foo08:
; DARWIN-32-DYNAMIC: 	movl	_lsrc, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _foo08:
; DARWIN-32-PIC: 	calll	L11$pb
; DARWIN-32-PIC-NEXT: L11$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lsrc-L11$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lptr-L11$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], ([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _foo08:
; DARWIN-64-STATIC: 	movl	_lsrc(%rip), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _foo08:
; DARWIN-64-DYNAMIC: 	movl	_lsrc(%rip), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _foo08:
; DARWIN-64-PIC: 	movl	_lsrc(%rip), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @qux00() nounwind {
entry:
	%0 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @src, i32 0, i64 16), align 4
	store i32 %0, i32* getelementptr ([131072 x i32], [131072 x i32]* @dst, i32 0, i64 16), align 4
	ret void
; LINUX-64-STATIC-LABEL: qux00:
; LINUX-64-STATIC: movl    src+64(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], dst+64(%rip)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: qux00:
; LINUX-32-STATIC: 	movl	src+64, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], dst+64
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: qux00:
; LINUX-32-PIC: 	movl	src+64, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], dst+64
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: qux00:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	dst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _qux00:
; DARWIN-32-STATIC: 	movl	_src+64, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _dst+64
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _qux00:
; DARWIN-32-DYNAMIC: 	movl	L_src$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	64([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_dst$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _qux00:
; DARWIN-32-PIC: 	calll	L12$pb
; DARWIN-32-PIC-NEXT: L12$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L12$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	64([[ECX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L12$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], 64([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _qux00:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _qux00:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _qux00:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @qxx00() nounwind {
entry:
	%0 = load i32, i32* getelementptr ([32 x i32], [32 x i32]* @xsrc, i32 0, i64 16), align 4
	store i32 %0, i32* getelementptr ([32 x i32], [32 x i32]* @xdst, i32 0, i64 16), align 4
	ret void
; LINUX-64-STATIC-LABEL: qxx00:
; LINUX-64-STATIC: movl    xsrc+64(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], xdst+64(%rip)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: qxx00:
; LINUX-32-STATIC: 	movl	xsrc+64, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], xdst+64
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: qxx00:
; LINUX-32-PIC: 	movl	xsrc+64, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], xdst+64
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: qxx00:
; LINUX-64-PIC: 	movq	xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _qxx00:
; DARWIN-32-STATIC: 	movl	_xsrc+64, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _xdst+64
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _qxx00:
; DARWIN-32-DYNAMIC: 	movl	L_xsrc$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	64([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_xdst$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _qxx00:
; DARWIN-32-PIC: 	calll	L13$pb
; DARWIN-32-PIC-NEXT: L13$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xsrc$non_lazy_ptr-L13$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	64([[ECX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xdst$non_lazy_ptr-L13$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], 64([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _qxx00:
; DARWIN-64-STATIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _qxx00:
; DARWIN-64-DYNAMIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _qxx00:
; DARWIN-64-PIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @qux01() nounwind {
entry:
	store i32* getelementptr ([131072 x i32], [131072 x i32]* @dst, i32 0, i64 16), i32** @ptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: qux01:
; LINUX-64-STATIC: movq    $dst+64, ptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: qux01:
; LINUX-32-STATIC: 	movl	$dst+64, ptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: qux01:
; LINUX-32-PIC: 	movl	$dst+64, ptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: qux01:
; LINUX-64-PIC: 	movq	dst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	addq	$64, [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _qux01:
; DARWIN-32-STATIC: 	movl	$_dst+64, _ptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _qux01:
; DARWIN-32-DYNAMIC: 	movl	L_dst$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	addl	$64, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _qux01:
; DARWIN-32-PIC: 	calll	L14$pb
; DARWIN-32-PIC-NEXT: L14$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L14$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	addl	$64, [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L14$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], ([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _qux01:
; DARWIN-64-STATIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	addq	$64, [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _qux01:
; DARWIN-64-DYNAMIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	addq	$64, [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _qux01:
; DARWIN-64-PIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	addq	$64, [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @qxx01() nounwind {
entry:
	store i32* getelementptr ([32 x i32], [32 x i32]* @xdst, i32 0, i64 16), i32** @ptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: qxx01:
; LINUX-64-STATIC: movq    $xdst+64, ptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: qxx01:
; LINUX-32-STATIC: 	movl	$xdst+64, ptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: qxx01:
; LINUX-32-PIC: 	movl	$xdst+64, ptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: qxx01:
; LINUX-64-PIC: 	movq	xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	addq	$64, [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _qxx01:
; DARWIN-32-STATIC: 	movl	$_xdst+64, _ptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _qxx01:
; DARWIN-32-DYNAMIC: 	movl	L_xdst$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	addl	$64, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _qxx01:
; DARWIN-32-PIC: 	calll	L15$pb
; DARWIN-32-PIC-NEXT: L15$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xdst$non_lazy_ptr-L15$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	addl	$64, [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L15$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], ([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _qxx01:
; DARWIN-64-STATIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	addq	$64, [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _qxx01:
; DARWIN-64-DYNAMIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	addq	$64, [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _qxx01:
; DARWIN-64-PIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	addq	$64, [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @qux02() nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @src, i32 0, i64 16), align 4
	%2 = getelementptr i32, i32* %0, i64 16
	store i32 %1, i32* %2, align 4
; LINUX-64-STATIC-LABEL: qux02:
; LINUX-64-STATIC: movl    src+64(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    ptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 64([[RCX]])
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: qux02:
; LINUX-32-STATIC: 	movl	src+64, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; LINUX-32-STATIC-NEXT: 	ret
	ret void

; LINUX-32-PIC-LABEL: qux02:
; LINUX-32-PIC: 	movl	src+64, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: qux02:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _qux02:
; DARWIN-32-STATIC: 	movl	_src+64, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_ptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _qux02:
; DARWIN-32-DYNAMIC: 	movl	L_src$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	64([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[ECX]]), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _qux02:
; DARWIN-32-PIC: 	calll	L16$pb
; DARWIN-32-PIC-NEXT: L16$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L16$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	64([[ECX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L16$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], 64([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _qux02:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _qux02:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _qux02:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @qxx02() nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = load i32, i32* getelementptr ([32 x i32], [32 x i32]* @xsrc, i32 0, i64 16), align 4
	%2 = getelementptr i32, i32* %0, i64 16
	store i32 %1, i32* %2, align 4
; LINUX-64-STATIC-LABEL: qxx02:
; LINUX-64-STATIC: movl    xsrc+64(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    ptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 64([[RCX]])
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: qxx02:
; LINUX-32-STATIC: 	movl	xsrc+64, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; LINUX-32-STATIC-NEXT: 	ret
	ret void

; LINUX-32-PIC-LABEL: qxx02:
; LINUX-32-PIC: 	movl	xsrc+64, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: qxx02:
; LINUX-64-PIC: 	movq	xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _qxx02:
; DARWIN-32-STATIC: 	movl	_xsrc+64, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_ptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _qxx02:
; DARWIN-32-DYNAMIC: 	movl	L_xsrc$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	64([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[ECX]]), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _qxx02:
; DARWIN-32-PIC: 	calll	L17$pb
; DARWIN-32-PIC-NEXT: L17$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xsrc$non_lazy_ptr-L17$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	64([[ECX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L17$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], 64([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _qxx02:
; DARWIN-64-STATIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _qxx02:
; DARWIN-64-DYNAMIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _qxx02:
; DARWIN-64-PIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @qux03() nounwind {
entry:
	%0 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @dsrc, i32 0, i64 16), align 32
	store i32 %0, i32* getelementptr ([131072 x i32], [131072 x i32]* @ddst, i32 0, i64 16), align 32
	ret void
; LINUX-64-STATIC-LABEL: qux03:
; LINUX-64-STATIC: movl    dsrc+64(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], ddst+64(%rip)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: qux03:
; LINUX-32-STATIC: 	movl	dsrc+64, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ddst+64
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: qux03:
; LINUX-32-PIC: 	movl	dsrc+64, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ddst+64
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: qux03:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ddst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _qux03:
; DARWIN-32-STATIC: 	movl	_dsrc+64, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _ddst+64
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _qux03:
; DARWIN-32-DYNAMIC: 	movl	_dsrc+64, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], _ddst+64
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _qux03:
; DARWIN-32-PIC: 	calll	L18$pb
; DARWIN-32-PIC-NEXT: L18$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_dsrc-L18$pb)+64([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], (_ddst-L18$pb)+64([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _qux03:
; DARWIN-64-STATIC: 	movl	_dsrc+64(%rip), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], _ddst+64(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _qux03:
; DARWIN-64-DYNAMIC: 	movl	_dsrc+64(%rip), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], _ddst+64(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _qux03:
; DARWIN-64-PIC: 	movl	_dsrc+64(%rip), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], _ddst+64(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @qux04() nounwind {
entry:
	store i32* getelementptr ([131072 x i32], [131072 x i32]* @ddst, i32 0, i64 16), i32** @dptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: qux04:
; LINUX-64-STATIC: movq    $ddst+64, dptr(%rip)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: qux04:
; LINUX-32-STATIC: 	movl	$ddst+64, dptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: qux04:
; LINUX-32-PIC: 	movl	$ddst+64, dptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: qux04:
; LINUX-64-PIC: 	movq	ddst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	addq	$64, [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	dptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _qux04:
; DARWIN-32-STATIC: 	movl	$_ddst+64, _dptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _qux04:
; DARWIN-32-DYNAMIC: 	movl	$_ddst+64, _dptr
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _qux04:
; DARWIN-32-PIC: 	calll	L19$pb
; DARWIN-32-PIC-NEXT: L19$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ddst-L19$pb)+64([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _dptr-L19$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _qux04:
; DARWIN-64-STATIC: 	leaq	_ddst+64(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _qux04:
; DARWIN-64-DYNAMIC: 	leaq	_ddst+64(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _qux04:
; DARWIN-64-PIC: 	leaq	_ddst+64(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @qux05() nounwind {
entry:
	%0 = load i32*, i32** @dptr, align 8
	%1 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @dsrc, i32 0, i64 16), align 32
	%2 = getelementptr i32, i32* %0, i64 16
	store i32 %1, i32* %2, align 4
; LINUX-64-STATIC-LABEL: qux05:
; LINUX-64-STATIC: movl    dsrc+64(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    dptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 64([[RCX]])
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: qux05:
; LINUX-32-STATIC: 	movl	dsrc+64, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; LINUX-32-STATIC-NEXT: 	ret
	ret void

; LINUX-32-PIC-LABEL: qux05:
; LINUX-32-PIC: 	movl	dsrc+64, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: qux05:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	dptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _qux05:
; DARWIN-32-STATIC: 	movl	_dsrc+64, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _qux05:
; DARWIN-32-DYNAMIC: 	movl	_dsrc+64, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _qux05:
; DARWIN-32-PIC: 	calll	L20$pb
; DARWIN-32-PIC-NEXT: L20$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_dsrc-L20$pb)+64([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dptr-L20$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], 64([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _qux05:
; DARWIN-64-STATIC: 	movl	_dsrc+64(%rip), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _qux05:
; DARWIN-64-DYNAMIC: 	movl	_dsrc+64(%rip), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _qux05:
; DARWIN-64-PIC: 	movl	_dsrc+64(%rip), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @qux06() nounwind {
entry:
	%0 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @lsrc, i32 0, i64 16), align 4
	store i32 %0, i32* getelementptr ([131072 x i32], [131072 x i32]* @ldst, i32 0, i64 16), align 4
	ret void
; LINUX-64-STATIC-LABEL: qux06:
; LINUX-64-STATIC: movl    lsrc+64(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], ldst+64
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: qux06:
; LINUX-32-STATIC: 	movl	lsrc+64, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ldst+64
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: qux06:
; LINUX-32-PIC: 	movl	lsrc+64, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ldst+64
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: qux06:
; LINUX-64-PIC: 	movl	lsrc+64(%rip), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ldst+64(%rip)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _qux06:
; DARWIN-32-STATIC: 	movl	_lsrc+64, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _ldst+64
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _qux06:
; DARWIN-32-DYNAMIC: 	movl	_lsrc+64, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], _ldst+64
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _qux06:
; DARWIN-32-PIC: 	calll	L21$pb
; DARWIN-32-PIC-NEXT: L21$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_lsrc-L21$pb)+64([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], (_ldst-L21$pb)+64([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _qux06:
; DARWIN-64-STATIC: 	movl	_lsrc+64(%rip), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], _ldst+64(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _qux06:
; DARWIN-64-DYNAMIC: 	movl	_lsrc+64(%rip), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], _ldst+64(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _qux06:
; DARWIN-64-PIC: 	movl	_lsrc+64(%rip), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], _ldst+64(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @qux07() nounwind {
entry:
	store i32* getelementptr ([131072 x i32], [131072 x i32]* @ldst, i32 0, i64 16), i32** @lptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: qux07:
; LINUX-64-STATIC: movq    $ldst+64, lptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: qux07:
; LINUX-32-STATIC: 	movl	$ldst+64, lptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: qux07:
; LINUX-32-PIC: 	movl	$ldst+64, lptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: qux07:
; LINUX-64-PIC: 	leaq	ldst+64(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], lptr(%rip)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _qux07:
; DARWIN-32-STATIC: 	movl	$_ldst+64, _lptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _qux07:
; DARWIN-32-DYNAMIC: 	movl	$_ldst+64, _lptr
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _qux07:
; DARWIN-32-PIC: 	calll	L22$pb
; DARWIN-32-PIC-NEXT: L22$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ldst-L22$pb)+64([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _lptr-L22$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _qux07:
; DARWIN-64-STATIC: 	leaq	_ldst+64(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _qux07:
; DARWIN-64-DYNAMIC: 	leaq	_ldst+64(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _qux07:
; DARWIN-64-PIC: 	leaq	_ldst+64(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @qux08() nounwind {
entry:
	%0 = load i32*, i32** @lptr, align 8
	%1 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @lsrc, i32 0, i64 16), align 4
	%2 = getelementptr i32, i32* %0, i64 16
	store i32 %1, i32* %2, align 4
; LINUX-64-STATIC-LABEL: qux08:
; LINUX-64-STATIC: movl    lsrc+64(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    lptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 64([[RCX]])
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: qux08:
; LINUX-32-STATIC: 	movl	lsrc+64, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; LINUX-32-STATIC-NEXT: 	ret
	ret void

; LINUX-32-PIC-LABEL: qux08:
; LINUX-32-PIC: 	movl	lsrc+64, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: qux08:
; LINUX-64-PIC: 	movl	lsrc+64(%rip), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	lptr(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _qux08:
; DARWIN-32-STATIC: 	movl	_lsrc+64, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _qux08:
; DARWIN-32-DYNAMIC: 	movl	_lsrc+64, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _qux08:
; DARWIN-32-PIC: 	calll	L23$pb
; DARWIN-32-PIC-NEXT: L23$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_lsrc-L23$pb)+64([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lptr-L23$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], 64([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _qux08:
; DARWIN-64-STATIC: 	movl	_lsrc+64(%rip), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _qux08:
; DARWIN-64-DYNAMIC: 	movl	_lsrc+64(%rip), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _qux08:
; DARWIN-64-PIC: 	movl	_lsrc+64(%rip), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ind00(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32], [131072 x i32]* @src, i64 0, i64 %i
	%1 = load i32, i32* %0, align 4
	%2 = getelementptr [131072 x i32], [131072 x i32]* @dst, i64 0, i64 %i
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC-LABEL: ind00:
; LINUX-64-STATIC: movl    src(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], dst(,%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ind00:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	src(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], dst(,[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ind00:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	src(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], dst(,[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ind00:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	dst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ind00:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_src(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], _dst(,[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ind00:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_src$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[ECX]],[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_dst$non_lazy_ptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ind00:
; DARWIN-32-PIC: 	calll	L24$pb
; DARWIN-32-PIC-NEXT: L24$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L24$pb([[EAX]]), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EDX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L24$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], ([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ind00:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ind00:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ind00:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ixd00(i64 %i) nounwind {
entry:
	%0 = getelementptr [32 x i32], [32 x i32]* @xsrc, i64 0, i64 %i
	%1 = load i32, i32* %0, align 4
	%2 = getelementptr [32 x i32], [32 x i32]* @xdst, i64 0, i64 %i
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC-LABEL: ixd00:
; LINUX-64-STATIC: movl    xsrc(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], xdst(,%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ixd00:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	xsrc(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], xdst(,[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ixd00:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	xsrc(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], xdst(,[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ixd00:
; LINUX-64-PIC: 	movq	xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ixd00:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_xsrc(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], _xdst(,[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ixd00:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_xsrc$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[ECX]],[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_xdst$non_lazy_ptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ixd00:
; DARWIN-32-PIC: 	calll	L25$pb
; DARWIN-32-PIC-NEXT: L25$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xsrc$non_lazy_ptr-L25$pb([[EAX]]), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EDX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xdst$non_lazy_ptr-L25$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], ([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ixd00:
; DARWIN-64-STATIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ixd00:
; DARWIN-64-DYNAMIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ixd00:
; DARWIN-64-PIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ind01(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32], [131072 x i32]* @dst, i64 0, i64 %i
	store i32* %0, i32** @ptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: ind01:
; LINUX-64-STATIC: leaq    dst(,%rdi,4), [[RAX:%r.x]]
; LINUX-64-STATIC: movq    [[RAX]], ptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ind01:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	dst(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ind01:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	dst(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ind01:
; LINUX-64-PIC: 	shlq	$2, %rdi
; LINUX-64-PIC-NEXT: 	addq	dst@GOTPCREL(%rip), %rdi
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	%rdi, ([[RAX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ind01:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_dst(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _ptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ind01:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	shll	$2, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	addl	L_dst$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ind01:
; DARWIN-32-PIC: 	calll	L26$pb
; DARWIN-32-PIC-NEXT: L26$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	shll	$2, [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	addl	L_dst$non_lazy_ptr-L26$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L26$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], ([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ind01:
; DARWIN-64-STATIC: 	shlq	$2, %rdi
; DARWIN-64-STATIC-NEXT: 	addq	_dst@GOTPCREL(%rip), %rdi
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	%rdi, ([[RAX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ind01:
; DARWIN-64-DYNAMIC: 	shlq	$2, %rdi
; DARWIN-64-DYNAMIC-NEXT: 	addq	_dst@GOTPCREL(%rip), %rdi
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	%rdi, ([[RAX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ind01:
; DARWIN-64-PIC: 	shlq	$2, %rdi
; DARWIN-64-PIC-NEXT: 	addq	_dst@GOTPCREL(%rip), %rdi
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	%rdi, ([[RAX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ixd01(i64 %i) nounwind {
entry:
	%0 = getelementptr [32 x i32], [32 x i32]* @xdst, i64 0, i64 %i
	store i32* %0, i32** @ptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: ixd01:
; LINUX-64-STATIC: leaq    xdst(,%rdi,4), [[RAX:%r.x]]
; LINUX-64-STATIC: movq    [[RAX]], ptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ixd01:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	xdst(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ixd01:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	xdst(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ixd01:
; LINUX-64-PIC: 	shlq	$2, %rdi
; LINUX-64-PIC-NEXT: 	addq	xdst@GOTPCREL(%rip), %rdi
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	%rdi, ([[RAX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ixd01:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_xdst(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _ptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ixd01:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	shll	$2, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	addl	L_xdst$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ixd01:
; DARWIN-32-PIC: 	calll	L27$pb
; DARWIN-32-PIC-NEXT: L27$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	shll	$2, [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	addl	L_xdst$non_lazy_ptr-L27$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L27$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], ([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ixd01:
; DARWIN-64-STATIC: 	shlq	$2, %rdi
; DARWIN-64-STATIC-NEXT: 	addq	_xdst@GOTPCREL(%rip), %rdi
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	%rdi, ([[RAX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ixd01:
; DARWIN-64-DYNAMIC: 	shlq	$2, %rdi
; DARWIN-64-DYNAMIC-NEXT: 	addq	_xdst@GOTPCREL(%rip), %rdi
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	%rdi, ([[RAX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ixd01:
; DARWIN-64-PIC: 	shlq	$2, %rdi
; DARWIN-64-PIC-NEXT: 	addq	_xdst@GOTPCREL(%rip), %rdi
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	%rdi, ([[RAX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ind02(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = getelementptr [131072 x i32], [131072 x i32]* @src, i64 0, i64 %i
	%2 = load i32, i32* %1, align 4
	%3 = getelementptr i32, i32* %0, i64 %i
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC-LABEL: ind02:
; LINUX-64-STATIC: movl    src(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    ptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], ([[RCX]],%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ind02:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	src(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	ptr, [[EDX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ind02:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	src(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	ptr, [[EDX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ind02:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ind02:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_src(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_ptr, [[EDX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ind02:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_src$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[ECX]],[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[EDX]]), [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ind02:
; DARWIN-32-PIC: 	calll	L28$pb
; DARWIN-32-PIC-NEXT: L28$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L28$pb([[EAX]]), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EDX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L28$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], ([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ind02:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ind02:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ind02:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ixd02(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = getelementptr [32 x i32], [32 x i32]* @xsrc, i64 0, i64 %i
	%2 = load i32, i32* %1, align 4
	%3 = getelementptr i32, i32* %0, i64 %i
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC-LABEL: ixd02:
; LINUX-64-STATIC: movl    xsrc(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    ptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], ([[RCX]],%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ixd02:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	xsrc(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	ptr, [[EDX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ixd02:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	xsrc(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	ptr, [[EDX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ixd02:
; LINUX-64-PIC: 	movq	xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ixd02:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_xsrc(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_ptr, [[EDX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ixd02:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_xsrc$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[ECX]],[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[EDX]]), [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ixd02:
; DARWIN-32-PIC: 	calll	L29$pb
; DARWIN-32-PIC-NEXT: L29$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xsrc$non_lazy_ptr-L29$pb([[EAX]]), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EDX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L29$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], ([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ixd02:
; DARWIN-64-STATIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ixd02:
; DARWIN-64-DYNAMIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ixd02:
; DARWIN-64-PIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ind03(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32], [131072 x i32]* @dsrc, i64 0, i64 %i
	%1 = load i32, i32* %0, align 4
	%2 = getelementptr [131072 x i32], [131072 x i32]* @ddst, i64 0, i64 %i
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC-LABEL: ind03:
; LINUX-64-STATIC: movl    dsrc(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], ddst(,%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ind03:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dsrc(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], ddst(,[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ind03:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dsrc(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], ddst(,[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ind03:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ddst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ind03:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dsrc(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], _ddst(,[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ind03:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dsrc(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], _ddst(,[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ind03:
; DARWIN-32-PIC: 	calll	L30$pb
; DARWIN-32-PIC-NEXT: L30$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dsrc-L30$pb([[EAX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], _ddst-L30$pb([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ind03:
; DARWIN-64-STATIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	_ddst(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ind03:
; DARWIN-64-DYNAMIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	_ddst(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ind03:
; DARWIN-64-PIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	leaq	_ddst(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ind04(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32], [131072 x i32]* @ddst, i64 0, i64 %i
	store i32* %0, i32** @dptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: ind04:
; LINUX-64-STATIC: leaq    ddst(,%rdi,4), [[RAX:%r.x]]
; LINUX-64-STATIC: movq    [[RAX]], dptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ind04:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	ddst(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], dptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ind04:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	ddst(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], dptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ind04:
; LINUX-64-PIC: 	shlq	$2, %rdi
; LINUX-64-PIC-NEXT: 	addq	ddst@GOTPCREL(%rip), %rdi
; LINUX-64-PIC-NEXT: 	movq	dptr@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	%rdi, ([[RAX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ind04:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_ddst(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _dptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ind04:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_ddst(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], _dptr
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ind04:
; DARWIN-32-PIC: 	calll	L31$pb
; DARWIN-32-PIC-NEXT: L31$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_ddst-L31$pb([[EAX]],[[ECX]],4), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _dptr-L31$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ind04:
; DARWIN-64-STATIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ind04:
; DARWIN-64-DYNAMIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ind04:
; DARWIN-64-PIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ind05(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @dptr, align 8
	%1 = getelementptr [131072 x i32], [131072 x i32]* @dsrc, i64 0, i64 %i
	%2 = load i32, i32* %1, align 4
	%3 = getelementptr i32, i32* %0, i64 %i
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC-LABEL: ind05:
; LINUX-64-STATIC: movl    dsrc(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    dptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], ([[RCX]],%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ind05:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dsrc(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dptr, [[EDX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ind05:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dsrc(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dptr, [[EDX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ind05:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	dptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ind05:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dsrc(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dptr, [[EDX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ind05:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dsrc(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ind05:
; DARWIN-32-PIC: 	calll	L32$pb
; DARWIN-32-PIC-NEXT: L32$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dsrc-L32$pb([[EAX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dptr-L32$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], ([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ind05:
; DARWIN-64-STATIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ind05:
; DARWIN-64-DYNAMIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ind05:
; DARWIN-64-PIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ind06(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32], [131072 x i32]* @lsrc, i64 0, i64 %i
	%1 = load i32, i32* %0, align 4
	%2 = getelementptr [131072 x i32], [131072 x i32]* @ldst, i64 0, i64 %i
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC-LABEL: ind06:
; LINUX-64-STATIC: movl    lsrc(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], ldst(,%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ind06:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lsrc(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], ldst(,[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ind06:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lsrc(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], ldst(,[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ind06:
; LINUX-64-PIC: 	leaq	lsrc(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	leaq	ldst(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ind06:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lsrc(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], _ldst(,[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ind06:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lsrc(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], _ldst(,[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ind06:
; DARWIN-32-PIC: 	calll	L33$pb
; DARWIN-32-PIC-NEXT: L33$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lsrc-L33$pb([[EAX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], _ldst-L33$pb([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ind06:
; DARWIN-64-STATIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	_ldst(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ind06:
; DARWIN-64-DYNAMIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	_ldst(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ind06:
; DARWIN-64-PIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	leaq	_ldst(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ind07(i64 %i) nounwind {
entry:
	%0 = getelementptr [131072 x i32], [131072 x i32]* @ldst, i64 0, i64 %i
	store i32* %0, i32** @lptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: ind07:
; LINUX-64-STATIC: leaq    ldst(,%rdi,4), [[RAX:%r.x]]
; LINUX-64-STATIC: movq    [[RAX]], lptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ind07:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	ldst(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], lptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ind07:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	ldst(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], lptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ind07:
; LINUX-64-PIC: 	leaq	ldst(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	([[RAX]],%rdi,4), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], lptr(%rip)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ind07:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_ldst(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _lptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ind07:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_ldst(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], _lptr
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ind07:
; DARWIN-32-PIC: 	calll	L34$pb
; DARWIN-32-PIC-NEXT: L34$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_ldst-L34$pb([[EAX]],[[ECX]],4), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _lptr-L34$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ind07:
; DARWIN-64-STATIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ind07:
; DARWIN-64-DYNAMIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ind07:
; DARWIN-64-PIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ind08(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @lptr, align 8
	%1 = getelementptr [131072 x i32], [131072 x i32]* @lsrc, i64 0, i64 %i
	%2 = load i32, i32* %1, align 4
	%3 = getelementptr i32, i32* %0, i64 %i
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC-LABEL: ind08:
; LINUX-64-STATIC: movl    lsrc(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    lptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], ([[RCX]],%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ind08:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lsrc(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lptr, [[EDX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ind08:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lsrc(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lptr, [[EDX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ind08:
; LINUX-64-PIC: 	leaq	lsrc(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	lptr(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ind08:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lsrc(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lptr, [[EDX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ind08:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lsrc(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], ([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ind08:
; DARWIN-32-PIC: 	calll	L35$pb
; DARWIN-32-PIC-NEXT: L35$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lsrc-L35$pb([[EAX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lptr-L35$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], ([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ind08:
; DARWIN-64-STATIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ind08:
; DARWIN-64-DYNAMIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ind08:
; DARWIN-64-PIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], ([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @off00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32], [131072 x i32]* @src, i64 0, i64 %0
	%2 = load i32, i32* %1, align 4
	%3 = getelementptr [131072 x i32], [131072 x i32]* @dst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC-LABEL: off00:
; LINUX-64-STATIC: movl    src+64(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], dst+64(,%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: off00:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	src+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], dst+64(,[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: off00:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	src+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], dst+64(,[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: off00:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	dst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _off00:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_src+64(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], _dst+64(,[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _off00:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_src$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	64([[ECX]],[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_dst$non_lazy_ptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _off00:
; DARWIN-32-PIC: 	calll	L36$pb
; DARWIN-32-PIC-NEXT: L36$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L36$pb([[EAX]]), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	64([[EDX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L36$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], 64([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _off00:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _off00:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _off00:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @oxf00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [32 x i32], [32 x i32]* @xsrc, i64 0, i64 %0
	%2 = load i32, i32* %1, align 4
	%3 = getelementptr [32 x i32], [32 x i32]* @xdst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC-LABEL: oxf00:
; LINUX-64-STATIC: movl    xsrc+64(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], xdst+64(,%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: oxf00:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	xsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], xdst+64(,[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: oxf00:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	xsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], xdst+64(,[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: oxf00:
; LINUX-64-PIC: 	movq	xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _oxf00:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_xsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], _xdst+64(,[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _oxf00:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_xsrc$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	64([[ECX]],[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_xdst$non_lazy_ptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _oxf00:
; DARWIN-32-PIC: 	calll	L37$pb
; DARWIN-32-PIC-NEXT: L37$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xsrc$non_lazy_ptr-L37$pb([[EAX]]), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	64([[EDX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xdst$non_lazy_ptr-L37$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], 64([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _oxf00:
; DARWIN-64-STATIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _oxf00:
; DARWIN-64-DYNAMIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _oxf00:
; DARWIN-64-PIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_xdst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @off01(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 16
	%0 = getelementptr [131072 x i32], [131072 x i32]* @dst, i64 0, i64 %.sum
	store i32* %0, i32** @ptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: off01:
; LINUX-64-STATIC: leaq    dst+64(,%rdi,4), [[RAX:%r.x]]
; LINUX-64-STATIC: movq    [[RAX]], ptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: off01:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	dst+64(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: off01:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	dst+64(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: off01:
; LINUX-64-PIC: 	movq	dst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _off01:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_dst+64(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _ptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _off01:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_dst$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _off01:
; DARWIN-32-PIC: 	calll	L38$pb
; DARWIN-32-PIC-NEXT: L38$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L38$pb([[EAX]]), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	64([[EDX]],[[ECX]],4), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L38$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], ([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _off01:
; DARWIN-64-STATIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _off01:
; DARWIN-64-DYNAMIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _off01:
; DARWIN-64-PIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @oxf01(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 16
	%0 = getelementptr [32 x i32], [32 x i32]* @xdst, i64 0, i64 %.sum
	store i32* %0, i32** @ptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: oxf01:
; LINUX-64-STATIC: leaq    xdst+64(,%rdi,4), [[RAX:%r.x]]
; LINUX-64-STATIC: movq    [[RAX]], ptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: oxf01:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	xdst+64(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: oxf01:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	xdst+64(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: oxf01:
; LINUX-64-PIC: 	movq	xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _oxf01:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_xdst+64(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _ptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _oxf01:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_xdst$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _oxf01:
; DARWIN-32-PIC: 	calll	L39$pb
; DARWIN-32-PIC-NEXT: L39$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xdst$non_lazy_ptr-L39$pb([[EAX]]), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	64([[EDX]],[[ECX]],4), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L39$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], ([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _oxf01:
; DARWIN-64-STATIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _oxf01:
; DARWIN-64-DYNAMIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _oxf01:
; DARWIN-64-PIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @off02(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr [131072 x i32], [131072 x i32]* @src, i64 0, i64 %1
	%3 = load i32, i32* %2, align 4
	%4 = getelementptr i32, i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC-LABEL: off02:
; LINUX-64-STATIC: movl    src+64(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    ptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 64([[RCX]],%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: off02:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	src+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	ptr, [[EDX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: off02:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	src+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	ptr, [[EDX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: off02:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _off02:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_src+64(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_ptr, [[EDX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _off02:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_src$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	64([[ECX]],[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[EDX]]), [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _off02:
; DARWIN-32-PIC: 	calll	L40$pb
; DARWIN-32-PIC-NEXT: L40$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L40$pb([[EAX]]), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	64([[EDX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L40$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], 64([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _off02:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _off02:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _off02:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @oxf02(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr [32 x i32], [32 x i32]* @xsrc, i64 0, i64 %1
	%3 = load i32, i32* %2, align 4
	%4 = getelementptr i32, i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC-LABEL: oxf02:
; LINUX-64-STATIC: movl    xsrc+64(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    ptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 64([[RCX]],%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: oxf02:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	xsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	ptr, [[EDX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: oxf02:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	xsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	ptr, [[EDX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: oxf02:
; LINUX-64-PIC: 	movq	xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _oxf02:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_xsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_ptr, [[EDX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _oxf02:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_xsrc$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	64([[ECX]],[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[EDX]]), [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _oxf02:
; DARWIN-32-PIC: 	calll	L41$pb
; DARWIN-32-PIC-NEXT: L41$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xsrc$non_lazy_ptr-L41$pb([[EAX]]), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	64([[EDX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L41$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], 64([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _oxf02:
; DARWIN-64-STATIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _oxf02:
; DARWIN-64-DYNAMIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _oxf02:
; DARWIN-64-PIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @off03(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32], [131072 x i32]* @dsrc, i64 0, i64 %0
	%2 = load i32, i32* %1, align 4
	%3 = getelementptr [131072 x i32], [131072 x i32]* @ddst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC-LABEL: off03:
; LINUX-64-STATIC: movl    dsrc+64(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], ddst+64(,%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: off03:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], ddst+64(,[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: off03:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], ddst+64(,[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: off03:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ddst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _off03:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], _ddst+64(,[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _off03:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], _ddst+64(,[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _off03:
; DARWIN-32-PIC: 	calll	L42$pb
; DARWIN-32-PIC-NEXT: L42$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_dsrc-L42$pb)+64([[EAX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], (_ddst-L42$pb)+64([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _off03:
; DARWIN-64-STATIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	_ddst(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _off03:
; DARWIN-64-DYNAMIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	_ddst(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _off03:
; DARWIN-64-PIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	leaq	_ddst(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @off04(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 16
	%0 = getelementptr [131072 x i32], [131072 x i32]* @ddst, i64 0, i64 %.sum
	store i32* %0, i32** @dptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: off04:
; LINUX-64-STATIC: leaq    ddst+64(,%rdi,4), [[RAX:%r.x]]
; LINUX-64-STATIC: movq    [[RAX]], dptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: off04:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	ddst+64(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], dptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: off04:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	ddst+64(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], dptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: off04:
; LINUX-64-PIC: 	movq	ddst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	dptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _off04:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_ddst+64(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _dptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _off04:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_ddst+64(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], _dptr
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _off04:
; DARWIN-32-PIC: 	calll	L43$pb
; DARWIN-32-PIC-NEXT: L43$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ddst-L43$pb)+64([[EAX]],[[ECX]],4), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _dptr-L43$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _off04:
; DARWIN-64-STATIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _off04:
; DARWIN-64-DYNAMIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _off04:
; DARWIN-64-PIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @off05(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @dptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr [131072 x i32], [131072 x i32]* @dsrc, i64 0, i64 %1
	%3 = load i32, i32* %2, align 4
	%4 = getelementptr i32, i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC-LABEL: off05:
; LINUX-64-STATIC: movl    dsrc+64(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    dptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 64([[RCX]],%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: off05:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dptr, [[EDX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: off05:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dptr, [[EDX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: off05:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	dptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _off05:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dptr, [[EDX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _off05:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _off05:
; DARWIN-32-PIC: 	calll	L44$pb
; DARWIN-32-PIC-NEXT: L44$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_dsrc-L44$pb)+64([[EAX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dptr-L44$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], 64([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _off05:
; DARWIN-64-STATIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _off05:
; DARWIN-64-DYNAMIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _off05:
; DARWIN-64-PIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @off06(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32], [131072 x i32]* @lsrc, i64 0, i64 %0
	%2 = load i32, i32* %1, align 4
	%3 = getelementptr [131072 x i32], [131072 x i32]* @ldst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC-LABEL: off06:
; LINUX-64-STATIC: movl    lsrc+64(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], ldst+64(,%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: off06:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], ldst+64(,[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: off06:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], ldst+64(,[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: off06:
; LINUX-64-PIC: 	leaq	lsrc(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	leaq	ldst(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _off06:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], _ldst+64(,[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _off06:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], _ldst+64(,[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _off06:
; DARWIN-32-PIC: 	calll	L45$pb
; DARWIN-32-PIC-NEXT: L45$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_lsrc-L45$pb)+64([[EAX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], (_ldst-L45$pb)+64([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _off06:
; DARWIN-64-STATIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	_ldst(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _off06:
; DARWIN-64-DYNAMIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	_ldst(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _off06:
; DARWIN-64-PIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	leaq	_ldst(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @off07(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 16
	%0 = getelementptr [131072 x i32], [131072 x i32]* @ldst, i64 0, i64 %.sum
	store i32* %0, i32** @lptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: off07:
; LINUX-64-STATIC: leaq    ldst+64(,%rdi,4), [[RAX:%r.x]]
; LINUX-64-STATIC: movq    [[RAX]], lptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: off07:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	ldst+64(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], lptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: off07:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	ldst+64(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], lptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: off07:
; LINUX-64-PIC: 	leaq	ldst(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], lptr(%rip)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _off07:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_ldst+64(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _lptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _off07:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_ldst+64(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], _lptr
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _off07:
; DARWIN-32-PIC: 	calll	L46$pb
; DARWIN-32-PIC-NEXT: L46$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ldst-L46$pb)+64([[EAX]],[[ECX]],4), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _lptr-L46$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _off07:
; DARWIN-64-STATIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _off07:
; DARWIN-64-DYNAMIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _off07:
; DARWIN-64-PIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @off08(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @lptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr [131072 x i32], [131072 x i32]* @lsrc, i64 0, i64 %1
	%3 = load i32, i32* %2, align 4
	%4 = getelementptr i32, i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC-LABEL: off08:
; LINUX-64-STATIC: movl    lsrc+64(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    lptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 64([[RCX]],%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: off08:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lptr, [[EDX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: off08:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lptr, [[EDX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: off08:
; LINUX-64-PIC: 	leaq	lsrc(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	lptr(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _off08:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lptr, [[EDX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _off08:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lsrc+64(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], 64([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _off08:
; DARWIN-32-PIC: 	calll	L47$pb
; DARWIN-32-PIC-NEXT: L47$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_lsrc-L47$pb)+64([[EAX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lptr-L47$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], 64([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _off08:
; DARWIN-64-STATIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _off08:
; DARWIN-64-DYNAMIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _off08:
; DARWIN-64-PIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	64([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 64([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @moo00(i64 %i) nounwind {
entry:
	%0 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @src, i32 0, i64 65536), align 4
	store i32 %0, i32* getelementptr ([131072 x i32], [131072 x i32]* @dst, i32 0, i64 65536), align 4
	ret void
; LINUX-64-STATIC-LABEL: moo00:
; LINUX-64-STATIC: movl    src+262144(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], dst+262144(%rip)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: moo00:
; LINUX-32-STATIC: 	movl	src+262144, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], dst+262144
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: moo00:
; LINUX-32-PIC: 	movl	src+262144, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], dst+262144
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: moo00:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	262144([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	dst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _moo00:
; DARWIN-32-STATIC: 	movl	_src+262144, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _dst+262144
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _moo00:
; DARWIN-32-DYNAMIC: 	movl	L_src$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	262144([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_dst$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _moo00:
; DARWIN-32-PIC: 	calll	L48$pb
; DARWIN-32-PIC-NEXT: L48$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L48$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	262144([[ECX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L48$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], 262144([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _moo00:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	262144([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _moo00:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	262144([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _moo00:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	262144([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @moo01(i64 %i) nounwind {
entry:
	store i32* getelementptr ([131072 x i32], [131072 x i32]* @dst, i32 0, i64 65536), i32** @ptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: moo01:
; LINUX-64-STATIC: movq    $dst+262144, ptr(%rip)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: moo01:
; LINUX-32-STATIC: 	movl	$dst+262144, ptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: moo01:
; LINUX-32-PIC: 	movl	$dst+262144, ptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: moo01:
; LINUX-64-PIC: 	movl	$262144, [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	addq	dst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _moo01:
; DARWIN-32-STATIC: 	movl	$_dst+262144, _ptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _moo01:
; DARWIN-32-DYNAMIC: 	movl	$262144, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	addl	L_dst$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _moo01:
; DARWIN-32-PIC: 	calll	L49$pb
; DARWIN-32-PIC-NEXT: L49$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	$262144, [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	addl	L_dst$non_lazy_ptr-L49$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L49$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], ([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _moo01:
; DARWIN-64-STATIC: 	movl	$262144, [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	addq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _moo01:
; DARWIN-64-DYNAMIC: 	movl	$262144, [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	addq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _moo01:
; DARWIN-64-PIC: 	movl	$262144, [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	addq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @moo02(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @src, i32 0, i64 65536), align 4
	%2 = getelementptr i32, i32* %0, i64 65536
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC-LABEL: moo02:
; LINUX-64-STATIC: movl    src+262144(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    ptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 262144([[RCX]])
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: moo02:
; LINUX-32-STATIC: 	movl	src+262144, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], 262144([[ECX]])
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: moo02:
; LINUX-32-PIC: 	movl	src+262144, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], 262144([[ECX]])
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: moo02:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	262144([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _moo02:
; DARWIN-32-STATIC: 	movl	_src+262144, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_ptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], 262144([[ECX]])
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _moo02:
; DARWIN-32-DYNAMIC: 	movl	L_src$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	262144([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[ECX]]), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _moo02:
; DARWIN-32-PIC: 	calll	L50$pb
; DARWIN-32-PIC-NEXT: L50$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L50$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	262144([[ECX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L50$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], 262144([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _moo02:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	262144([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _moo02:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	262144([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _moo02:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	262144([[RAX]]), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @moo03(i64 %i) nounwind {
entry:
	%0 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @dsrc, i32 0, i64 65536), align 32
	store i32 %0, i32* getelementptr ([131072 x i32], [131072 x i32]* @ddst, i32 0, i64 65536), align 32
	ret void
; LINUX-64-STATIC-LABEL: moo03:
; LINUX-64-STATIC: movl    dsrc+262144(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], ddst+262144(%rip)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: moo03:
; LINUX-32-STATIC: 	movl	dsrc+262144, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ddst+262144
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: moo03:
; LINUX-32-PIC: 	movl	dsrc+262144, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ddst+262144
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: moo03:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	262144([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ddst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _moo03:
; DARWIN-32-STATIC: 	movl	_dsrc+262144, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _ddst+262144
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _moo03:
; DARWIN-32-DYNAMIC: 	movl	_dsrc+262144, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], _ddst+262144
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _moo03:
; DARWIN-32-PIC: 	calll	L51$pb
; DARWIN-32-PIC-NEXT: L51$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_dsrc-L51$pb)+262144([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], (_ddst-L51$pb)+262144([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _moo03:
; DARWIN-64-STATIC: 	movl	_dsrc+262144(%rip), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], _ddst+262144(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _moo03:
; DARWIN-64-DYNAMIC: 	movl	_dsrc+262144(%rip), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], _ddst+262144(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _moo03:
; DARWIN-64-PIC: 	movl	_dsrc+262144(%rip), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], _ddst+262144(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @moo04(i64 %i) nounwind {
entry:
	store i32* getelementptr ([131072 x i32], [131072 x i32]* @ddst, i32 0, i64 65536), i32** @dptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: moo04:
; LINUX-64-STATIC: movq    $ddst+262144, dptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: moo04:
; LINUX-32-STATIC: 	movl	$ddst+262144, dptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: moo04:
; LINUX-32-PIC: 	movl	$ddst+262144, dptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: moo04:
; LINUX-64-PIC: 	movl	$262144, [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	addq	ddst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	dptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _moo04:
; DARWIN-32-STATIC: 	movl	$_ddst+262144, _dptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _moo04:
; DARWIN-32-DYNAMIC: 	movl	$_ddst+262144, _dptr
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _moo04:
; DARWIN-32-PIC: 	calll	L52$pb
; DARWIN-32-PIC-NEXT: L52$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ddst-L52$pb)+262144([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _dptr-L52$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _moo04:
; DARWIN-64-STATIC: 	leaq	_ddst+262144(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _moo04:
; DARWIN-64-DYNAMIC: 	leaq	_ddst+262144(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _moo04:
; DARWIN-64-PIC: 	leaq	_ddst+262144(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @moo05(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @dptr, align 8
	%1 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @dsrc, i32 0, i64 65536), align 32
	%2 = getelementptr i32, i32* %0, i64 65536
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC-LABEL: moo05:
; LINUX-64-STATIC: movl    dsrc+262144(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    dptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 262144([[RCX]])
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: moo05:
; LINUX-32-STATIC: 	movl	dsrc+262144, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], 262144([[ECX]])
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: moo05:
; LINUX-32-PIC: 	movl	dsrc+262144, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], 262144([[ECX]])
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: moo05:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	262144([[RAX]]), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	dptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _moo05:
; DARWIN-32-STATIC: 	movl	_dsrc+262144, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], 262144([[ECX]])
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _moo05:
; DARWIN-32-DYNAMIC: 	movl	_dsrc+262144, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _moo05:
; DARWIN-32-PIC: 	calll	L53$pb
; DARWIN-32-PIC-NEXT: L53$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_dsrc-L53$pb)+262144([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dptr-L53$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], 262144([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _moo05:
; DARWIN-64-STATIC: 	movl	_dsrc+262144(%rip), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _moo05:
; DARWIN-64-DYNAMIC: 	movl	_dsrc+262144(%rip), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _moo05:
; DARWIN-64-PIC: 	movl	_dsrc+262144(%rip), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @moo06(i64 %i) nounwind {
entry:
	%0 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @lsrc, i32 0, i64 65536), align 4
	store i32 %0, i32* getelementptr ([131072 x i32], [131072 x i32]* @ldst, i32 0, i64 65536), align 4
	ret void
; LINUX-64-STATIC-LABEL: moo06:
; LINUX-64-STATIC: movl    lsrc+262144(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], ldst+262144(%rip)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: moo06:
; LINUX-32-STATIC: 	movl	lsrc+262144, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ldst+262144
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: moo06:
; LINUX-32-PIC: 	movl	lsrc+262144, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ldst+262144
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: moo06:
; LINUX-64-PIC: 	movl	lsrc+262144(%rip), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], ldst+262144(%rip)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _moo06:
; DARWIN-32-STATIC: 	movl	_lsrc+262144, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _ldst+262144
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _moo06:
; DARWIN-32-DYNAMIC: 	movl	_lsrc+262144, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], _ldst+262144
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _moo06:
; DARWIN-32-PIC: 	calll	L54$pb
; DARWIN-32-PIC-NEXT: L54$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_lsrc-L54$pb)+262144([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], (_ldst-L54$pb)+262144([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _moo06:
; DARWIN-64-STATIC: 	movl	_lsrc+262144(%rip), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], _ldst+262144(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _moo06:
; DARWIN-64-DYNAMIC: 	movl	_lsrc+262144(%rip), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], _ldst+262144(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _moo06:
; DARWIN-64-PIC: 	movl	_lsrc+262144(%rip), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], _ldst+262144(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @moo07(i64 %i) nounwind {
entry:
	store i32* getelementptr ([131072 x i32], [131072 x i32]* @ldst, i32 0, i64 65536), i32** @lptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: moo07:
; LINUX-64-STATIC: movq    $ldst+262144, lptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: moo07:
; LINUX-32-STATIC: 	movl	$ldst+262144, lptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: moo07:
; LINUX-32-PIC: 	movl	$ldst+262144, lptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: moo07:
; LINUX-64-PIC: 	leaq	ldst+262144(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], lptr(%rip)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _moo07:
; DARWIN-32-STATIC: 	movl	$_ldst+262144, _lptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _moo07:
; DARWIN-32-DYNAMIC: 	movl	$_ldst+262144, _lptr
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _moo07:
; DARWIN-32-PIC: 	calll	L55$pb
; DARWIN-32-PIC-NEXT: L55$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ldst-L55$pb)+262144([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _lptr-L55$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _moo07:
; DARWIN-64-STATIC: 	leaq	_ldst+262144(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _moo07:
; DARWIN-64-DYNAMIC: 	leaq	_ldst+262144(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _moo07:
; DARWIN-64-PIC: 	leaq	_ldst+262144(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @moo08(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @lptr, align 8
	%1 = load i32, i32* getelementptr ([131072 x i32], [131072 x i32]* @lsrc, i32 0, i64 65536), align 4
	%2 = getelementptr i32, i32* %0, i64 65536
	store i32 %1, i32* %2, align 4
	ret void
; LINUX-64-STATIC-LABEL: moo08:
; LINUX-64-STATIC: movl    lsrc+262144(%rip), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    lptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 262144([[RCX]])
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: moo08:
; LINUX-32-STATIC: 	movl	lsrc+262144, [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], 262144([[ECX]])
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: moo08:
; LINUX-32-PIC: 	movl	lsrc+262144, [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], 262144([[ECX]])
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: moo08:
; LINUX-64-PIC: 	movl	lsrc+262144(%rip), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	lptr(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _moo08:
; DARWIN-32-STATIC: 	movl	_lsrc+262144, [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], 262144([[ECX]])
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _moo08:
; DARWIN-32-DYNAMIC: 	movl	_lsrc+262144, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _moo08:
; DARWIN-32-PIC: 	calll	L56$pb
; DARWIN-32-PIC-NEXT: L56$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_lsrc-L56$pb)+262144([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lptr-L56$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], 262144([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _moo08:
; DARWIN-64-STATIC: 	movl	_lsrc+262144(%rip), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _moo08:
; DARWIN-64-DYNAMIC: 	movl	_lsrc+262144(%rip), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _moo08:
; DARWIN-64-PIC: 	movl	_lsrc+262144(%rip), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @big00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32], [131072 x i32]* @src, i64 0, i64 %0
	%2 = load i32, i32* %1, align 4
	%3 = getelementptr [131072 x i32], [131072 x i32]* @dst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC-LABEL: big00:
; LINUX-64-STATIC: movl    src+262144(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], dst+262144(,%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: big00:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	src+262144(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], dst+262144(,[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: big00:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	src+262144(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], dst+262144(,[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: big00:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	dst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _big00:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_src+262144(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], _dst+262144(,[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _big00:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_src$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	262144([[ECX]],[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_dst$non_lazy_ptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], 262144([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _big00:
; DARWIN-32-PIC: 	calll	L57$pb
; DARWIN-32-PIC-NEXT: L57$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L57$pb([[EAX]]), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	262144([[EDX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L57$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], 262144([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _big00:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _big00:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _big00:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_dst@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @big01(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 65536
	%0 = getelementptr [131072 x i32], [131072 x i32]* @dst, i64 0, i64 %.sum
	store i32* %0, i32** @ptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: big01:
; LINUX-64-STATIC: leaq    dst+262144(,%rdi,4), [[RAX:%r.x]]
; LINUX-64-STATIC: movq    [[RAX]], ptr(%rip)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: big01:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	dst+262144(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], ptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: big01:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	dst+262144(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], ptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: big01:
; LINUX-64-PIC: 	movq	dst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _big01:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_dst+262144(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _ptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _big01:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_dst$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], ([[ECX]])
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _big01:
; DARWIN-32-PIC: 	calll	L58$pb
; DARWIN-32-PIC-NEXT: L58$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L58$pb([[EAX]]), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	262144([[EDX]],[[ECX]],4), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L58$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], ([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _big01:
; DARWIN-64-STATIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _big01:
; DARWIN-64-DYNAMIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _big01:
; DARWIN-64-PIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; DARWIN-64-PIC-NEXT: 	ret
}

define void @big02(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr [131072 x i32], [131072 x i32]* @src, i64 0, i64 %1
	%3 = load i32, i32* %2, align 4
	%4 = getelementptr i32, i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC-LABEL: big02:
; LINUX-64-STATIC: movl    src+262144(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    ptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 262144([[RCX]],%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: big02:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	src+262144(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	ptr, [[EDX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], 262144([[EDX]],[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: big02:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	src+262144(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	ptr, [[EDX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], 262144([[EDX]],[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: big02:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _big02:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_src+262144(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_ptr, [[EDX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], 262144([[EDX]],[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _big02:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_src$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	262144([[ECX]],[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ptr$non_lazy_ptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[EDX]]), [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], 262144([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _big02:
; DARWIN-32-PIC: 	calll	L59$pb
; DARWIN-32-PIC-NEXT: L59$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L59$pb([[EAX]]), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	262144([[EDX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L59$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], 262144([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _big02:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _big02:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _big02:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @big03(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32], [131072 x i32]* @dsrc, i64 0, i64 %0
	%2 = load i32, i32* %1, align 4
	%3 = getelementptr [131072 x i32], [131072 x i32]* @ddst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC-LABEL: big03:
; LINUX-64-STATIC: movl    dsrc+262144(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], ddst+262144(,%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: big03:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], ddst+262144(,[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: big03:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], ddst+262144(,[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: big03:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	ddst@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _big03:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], _ddst+262144(,[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _big03:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], _ddst+262144(,[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _big03:
; DARWIN-32-PIC: 	calll	L60$pb
; DARWIN-32-PIC-NEXT: L60$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_dsrc-L60$pb)+262144([[EAX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], (_ddst-L60$pb)+262144([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _big03:
; DARWIN-64-STATIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	_ddst(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _big03:
; DARWIN-64-DYNAMIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	_ddst(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _big03:
; DARWIN-64-PIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	leaq	_ddst(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @big04(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 65536
	%0 = getelementptr [131072 x i32], [131072 x i32]* @ddst, i64 0, i64 %.sum
	store i32* %0, i32** @dptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: big04:
; LINUX-64-STATIC: leaq    ddst+262144(,%rdi,4), [[RAX:%r.x]]
; LINUX-64-STATIC: movq    [[RAX]], dptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: big04:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	ddst+262144(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], dptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: big04:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	ddst+262144(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], dptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: big04:
; LINUX-64-PIC: 	movq	ddst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	dptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], ([[RCX]])
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _big04:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_ddst+262144(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _dptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _big04:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_ddst+262144(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], _dptr
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _big04:
; DARWIN-32-PIC: 	calll	L61$pb
; DARWIN-32-PIC-NEXT: L61$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ddst-L61$pb)+262144([[EAX]],[[ECX]],4), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _dptr-L61$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _big04:
; DARWIN-64-STATIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _big04:
; DARWIN-64-DYNAMIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _big04:
; DARWIN-64-PIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], _dptr(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @big05(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @dptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr [131072 x i32], [131072 x i32]* @dsrc, i64 0, i64 %1
	%3 = load i32, i32* %2, align 4
	%4 = getelementptr i32, i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC-LABEL: big05:
; LINUX-64-STATIC: movl    dsrc+262144(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    dptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 262144([[RCX]],%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: big05:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dptr, [[EDX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], 262144([[EDX]],[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: big05:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dptr, [[EDX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], 262144([[EDX]],[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: big05:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	dptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RCX]]), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _big05:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dptr, [[EDX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], 262144([[EDX]],[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _big05:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], 262144([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _big05:
; DARWIN-32-PIC: 	calll	L62$pb
; DARWIN-32-PIC-NEXT: L62$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_dsrc-L62$pb)+262144([[EAX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dptr-L62$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], 262144([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _big05:
; DARWIN-64-STATIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _big05:
; DARWIN-64-DYNAMIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _big05:
; DARWIN-64-PIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_dptr(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @big06(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32], [131072 x i32]* @lsrc, i64 0, i64 %0
	%2 = load i32, i32* %1, align 4
	%3 = getelementptr [131072 x i32], [131072 x i32]* @ldst, i64 0, i64 %0
	store i32 %2, i32* %3, align 4
	ret void
; LINUX-64-STATIC-LABEL: big06:
; LINUX-64-STATIC: movl    lsrc+262144(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movl    [[EAX]], ldst+262144(,%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: big06:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], ldst+262144(,[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: big06:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], ldst+262144(,[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: big06:
; LINUX-64-PIC: 	leaq	lsrc(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	leaq	ldst(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _big06:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], _ldst+262144(,[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _big06:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], _ldst+262144(,[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _big06:
; DARWIN-32-PIC: 	calll	L63$pb
; DARWIN-32-PIC-NEXT: L63$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_lsrc-L63$pb)+262144([[EAX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], (_ldst-L63$pb)+262144([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _big06:
; DARWIN-64-STATIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	_ldst(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _big06:
; DARWIN-64-DYNAMIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	_ldst(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _big06:
; DARWIN-64-PIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	leaq	_ldst(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @big07(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 65536
	%0 = getelementptr [131072 x i32], [131072 x i32]* @ldst, i64 0, i64 %.sum
	store i32* %0, i32** @lptr, align 8
	ret void
; LINUX-64-STATIC-LABEL: big07:
; LINUX-64-STATIC: leaq    ldst+262144(,%rdi,4), [[RAX:%r.x]]
; LINUX-64-STATIC: movq    [[RAX]], lptr
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: big07:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	ldst+262144(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[EAX]], lptr
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: big07:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	ldst+262144(,[[EAX]],4), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[EAX]], lptr
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: big07:
; LINUX-64-PIC: 	leaq	ldst(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	[[RAX]], lptr(%rip)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _big07:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_ldst+262144(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[EAX]], _lptr
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _big07:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_ldst+262144(,[[EAX]],4), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[EAX]], _lptr
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _big07:
; DARWIN-32-PIC: 	calll	L64$pb
; DARWIN-32-PIC-NEXT: L64$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ldst-L64$pb)+262144([[EAX]],[[ECX]],4), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[ECX]], _lptr-L64$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _big07:
; DARWIN-64-STATIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _big07:
; DARWIN-64-DYNAMIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _big07:
; DARWIN-64-PIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	[[RAX]], _lptr(%rip)
; DARWIN-64-PIC-NEXT: 	ret
}

define void @big08(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @lptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr [131072 x i32], [131072 x i32]* @lsrc, i64 0, i64 %1
	%3 = load i32, i32* %2, align 4
	%4 = getelementptr i32, i32* %0, i64 %1
	store i32 %3, i32* %4, align 4
	ret void
; LINUX-64-STATIC-LABEL: big08:
; LINUX-64-STATIC: movl    lsrc+262144(,%rdi,4), [[EAX:%e.x]]
; LINUX-64-STATIC: movq    lptr(%rip), [[RCX:%r.x]]
; LINUX-64-STATIC: movl    [[EAX]], 262144([[RCX]],%rdi,4)
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: big08:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lptr, [[EDX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	[[ECX]], 262144([[EDX]],[[EAX]],4)
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: big08:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lptr, [[EDX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	[[ECX]], 262144([[EDX]],[[EAX]],4)
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: big08:
; LINUX-64-PIC: 	leaq	lsrc(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; LINUX-64-PIC-NEXT: 	movq	lptr(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _big08:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lptr, [[EDX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	[[ECX]], 262144([[EDX]],[[EAX]],4)
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _big08:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lsrc+262144(,[[EAX]],4), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lptr, [[EDX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	[[ECX]], 262144([[EDX]],[[EAX]],4)
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _big08:
; DARWIN-32-PIC: 	calll	L65$pb
; DARWIN-32-PIC-NEXT: L65$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	(_lsrc-L65$pb)+262144([[EAX]],[[ECX]],4), [[EDX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lptr-L65$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	[[EDX]], 262144([[EAX]],[[ECX]],4)
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _big08:
; DARWIN-64-STATIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _big08:
; DARWIN-64-DYNAMIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _big08:
; DARWIN-64-PIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	262144([[RAX]],%rdi,4), [[EAX:%e.x]]
; DARWIN-64-PIC-NEXT: 	movq	_lptr(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movl	[[EAX]], 262144([[RCX]],%rdi,4)
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bar00() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @src to i8*)
; LINUX-64-STATIC-LABEL: bar00:
; LINUX-64-STATIC: movl    $src, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bar00:
; LINUX-32-STATIC: 	movl	$src, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bar00:
; LINUX-32-PIC: 	movl	$src, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bar00:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bar00:
; DARWIN-32-STATIC: 	movl	$_src, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bar00:
; DARWIN-32-DYNAMIC: 	movl	L_src$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bar00:
; DARWIN-32-PIC: 	calll	L66$pb
; DARWIN-32-PIC-NEXT: L66$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L66$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bar00:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bar00:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bar00:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bxr00() nounwind {
entry:
	ret i8* bitcast ([32 x i32]* @xsrc to i8*)
; LINUX-64-STATIC-LABEL: bxr00:
; LINUX-64-STATIC: movl    $xsrc, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bxr00:
; LINUX-32-STATIC: 	movl	$xsrc, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bxr00:
; LINUX-32-PIC: 	movl	$xsrc, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bxr00:
; LINUX-64-PIC: 	movq	xsrc@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bxr00:
; DARWIN-32-STATIC: 	movl	$_xsrc, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bxr00:
; DARWIN-32-DYNAMIC: 	movl	L_xsrc$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bxr00:
; DARWIN-32-PIC: 	calll	L67$pb
; DARWIN-32-PIC-NEXT: L67$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xsrc$non_lazy_ptr-L67$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bxr00:
; DARWIN-64-STATIC: 	movq	_xsrc@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bxr00:
; DARWIN-64-DYNAMIC: 	movq	_xsrc@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bxr00:
; DARWIN-64-PIC: 	movq	_xsrc@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bar01() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @dst to i8*)
; LINUX-64-STATIC-LABEL: bar01:
; LINUX-64-STATIC: movl    $dst, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bar01:
; LINUX-32-STATIC: 	movl	$dst, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bar01:
; LINUX-32-PIC: 	movl	$dst, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bar01:
; LINUX-64-PIC: 	movq	dst@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bar01:
; DARWIN-32-STATIC: 	movl	$_dst, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bar01:
; DARWIN-32-DYNAMIC: 	movl	L_dst$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bar01:
; DARWIN-32-PIC: 	calll	L68$pb
; DARWIN-32-PIC-NEXT: L68$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L68$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bar01:
; DARWIN-64-STATIC: 	movq	_dst@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bar01:
; DARWIN-64-DYNAMIC: 	movq	_dst@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bar01:
; DARWIN-64-PIC: 	movq	_dst@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bxr01() nounwind {
entry:
	ret i8* bitcast ([32 x i32]* @xdst to i8*)
; LINUX-64-STATIC-LABEL: bxr01:
; LINUX-64-STATIC: movl    $xdst, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bxr01:
; LINUX-32-STATIC: 	movl	$xdst, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bxr01:
; LINUX-32-PIC: 	movl	$xdst, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bxr01:
; LINUX-64-PIC: 	movq	xdst@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bxr01:
; DARWIN-32-STATIC: 	movl	$_xdst, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bxr01:
; DARWIN-32-DYNAMIC: 	movl	L_xdst$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bxr01:
; DARWIN-32-PIC: 	calll	L69$pb
; DARWIN-32-PIC-NEXT: L69$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xdst$non_lazy_ptr-L69$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bxr01:
; DARWIN-64-STATIC: 	movq	_xdst@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bxr01:
; DARWIN-64-DYNAMIC: 	movq	_xdst@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bxr01:
; DARWIN-64-PIC: 	movq	_xdst@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bar02() nounwind {
entry:
	ret i8* bitcast (i32** @ptr to i8*)
; LINUX-64-STATIC-LABEL: bar02:
; LINUX-64-STATIC: movl    $ptr, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bar02:
; LINUX-32-STATIC: 	movl	$ptr, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bar02:
; LINUX-32-PIC: 	movl	$ptr, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bar02:
; LINUX-64-PIC: 	movq	ptr@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bar02:
; DARWIN-32-STATIC: 	movl	$_ptr, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bar02:
; DARWIN-32-DYNAMIC: 	movl	L_ptr$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bar02:
; DARWIN-32-PIC: 	calll	L70$pb
; DARWIN-32-PIC-NEXT: L70$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L70$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bar02:
; DARWIN-64-STATIC: 	movq	_ptr@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bar02:
; DARWIN-64-DYNAMIC: 	movq	_ptr@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bar02:
; DARWIN-64-PIC: 	movq	_ptr@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bar03() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @dsrc to i8*)
; LINUX-64-STATIC-LABEL: bar03:
; LINUX-64-STATIC: movl    $dsrc, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bar03:
; LINUX-32-STATIC: 	movl	$dsrc, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bar03:
; LINUX-32-PIC: 	movl	$dsrc, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bar03:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bar03:
; DARWIN-32-STATIC: 	movl	$_dsrc, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bar03:
; DARWIN-32-DYNAMIC: 	movl	$_dsrc, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bar03:
; DARWIN-32-PIC: 	calll	L71$pb
; DARWIN-32-PIC-NEXT: L71$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_dsrc-L71$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bar03:
; DARWIN-64-STATIC: 	leaq	_dsrc(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bar03:
; DARWIN-64-DYNAMIC: 	leaq	_dsrc(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bar03:
; DARWIN-64-PIC: 	leaq	_dsrc(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bar04() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @ddst to i8*)
; LINUX-64-STATIC-LABEL: bar04:
; LINUX-64-STATIC: movl    $ddst, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bar04:
; LINUX-32-STATIC: 	movl	$ddst, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bar04:
; LINUX-32-PIC: 	movl	$ddst, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bar04:
; LINUX-64-PIC: 	movq	ddst@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bar04:
; DARWIN-32-STATIC: 	movl	$_ddst, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bar04:
; DARWIN-32-DYNAMIC: 	movl	$_ddst, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bar04:
; DARWIN-32-PIC: 	calll	L72$pb
; DARWIN-32-PIC-NEXT: L72$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_ddst-L72$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bar04:
; DARWIN-64-STATIC: 	leaq	_ddst(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bar04:
; DARWIN-64-DYNAMIC: 	leaq	_ddst(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bar04:
; DARWIN-64-PIC: 	leaq	_ddst(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bar05() nounwind {
entry:
	ret i8* bitcast (i32** @dptr to i8*)
; LINUX-64-STATIC-LABEL: bar05:
; LINUX-64-STATIC: movl    $dptr, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bar05:
; LINUX-32-STATIC: 	movl	$dptr, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bar05:
; LINUX-32-PIC: 	movl	$dptr, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bar05:
; LINUX-64-PIC: 	movq	dptr@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bar05:
; DARWIN-32-STATIC: 	movl	$_dptr, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bar05:
; DARWIN-32-DYNAMIC: 	movl	$_dptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bar05:
; DARWIN-32-PIC: 	calll	L73$pb
; DARWIN-32-PIC-NEXT: L73$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_dptr-L73$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bar05:
; DARWIN-64-STATIC: 	leaq	_dptr(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bar05:
; DARWIN-64-DYNAMIC: 	leaq	_dptr(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bar05:
; DARWIN-64-PIC: 	leaq	_dptr(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bar06() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @lsrc to i8*)
; LINUX-64-STATIC-LABEL: bar06:
; LINUX-64-STATIC: movl    $lsrc, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bar06:
; LINUX-32-STATIC: 	movl	$lsrc, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bar06:
; LINUX-32-PIC: 	movl	$lsrc, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bar06:
; LINUX-64-PIC: 	leaq	lsrc(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bar06:
; DARWIN-32-STATIC: 	movl	$_lsrc, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bar06:
; DARWIN-32-DYNAMIC: 	movl	$_lsrc, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bar06:
; DARWIN-32-PIC: 	calll	L74$pb
; DARWIN-32-PIC-NEXT: L74$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_lsrc-L74$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bar06:
; DARWIN-64-STATIC: 	leaq	_lsrc(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bar06:
; DARWIN-64-DYNAMIC: 	leaq	_lsrc(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bar06:
; DARWIN-64-PIC: 	leaq	_lsrc(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bar07() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @ldst to i8*)
; LINUX-64-STATIC-LABEL: bar07:
; LINUX-64-STATIC: movl    $ldst, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bar07:
; LINUX-32-STATIC: 	movl	$ldst, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bar07:
; LINUX-32-PIC: 	movl	$ldst, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bar07:
; LINUX-64-PIC: 	leaq	ldst(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bar07:
; DARWIN-32-STATIC: 	movl	$_ldst, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bar07:
; DARWIN-32-DYNAMIC: 	movl	$_ldst, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bar07:
; DARWIN-32-PIC: 	calll	L75$pb
; DARWIN-32-PIC-NEXT: L75$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_ldst-L75$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bar07:
; DARWIN-64-STATIC: 	leaq	_ldst(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bar07:
; DARWIN-64-DYNAMIC: 	leaq	_ldst(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bar07:
; DARWIN-64-PIC: 	leaq	_ldst(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bar08() nounwind {
entry:
	ret i8* bitcast (i32** @lptr to i8*)
; LINUX-64-STATIC-LABEL: bar08:
; LINUX-64-STATIC: movl    $lptr, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bar08:
; LINUX-32-STATIC: 	movl	$lptr, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bar08:
; LINUX-32-PIC: 	movl	$lptr, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bar08:
; LINUX-64-PIC: 	leaq	lptr(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bar08:
; DARWIN-32-STATIC: 	movl	$_lptr, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bar08:
; DARWIN-32-DYNAMIC: 	movl	$_lptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bar08:
; DARWIN-32-PIC: 	calll	L76$pb
; DARWIN-32-PIC-NEXT: L76$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_lptr-L76$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bar08:
; DARWIN-64-STATIC: 	leaq	_lptr(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bar08:
; DARWIN-64-DYNAMIC: 	leaq	_lptr(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bar08:
; DARWIN-64-PIC: 	leaq	_lptr(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @har00() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @src to i8*)
; LINUX-64-STATIC-LABEL: har00:
; LINUX-64-STATIC: movl    $src, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: har00:
; LINUX-32-STATIC: 	movl	$src, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: har00:
; LINUX-32-PIC: 	movl	$src, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: har00:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _har00:
; DARWIN-32-STATIC: 	movl	$_src, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _har00:
; DARWIN-32-DYNAMIC: 	movl	L_src$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _har00:
; DARWIN-32-PIC: 	calll	L77$pb
; DARWIN-32-PIC-NEXT: L77$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L77$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _har00:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _har00:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _har00:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @hxr00() nounwind {
entry:
	ret i8* bitcast ([32 x i32]* @xsrc to i8*)
; LINUX-64-STATIC-LABEL: hxr00:
; LINUX-64-STATIC: movl    $xsrc, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: hxr00:
; LINUX-32-STATIC: 	movl	$xsrc, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: hxr00:
; LINUX-32-PIC: 	movl	$xsrc, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: hxr00:
; LINUX-64-PIC: 	movq	xsrc@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _hxr00:
; DARWIN-32-STATIC: 	movl	$_xsrc, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _hxr00:
; DARWIN-32-DYNAMIC: 	movl	L_xsrc$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _hxr00:
; DARWIN-32-PIC: 	calll	L78$pb
; DARWIN-32-PIC-NEXT: L78$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xsrc$non_lazy_ptr-L78$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _hxr00:
; DARWIN-64-STATIC: 	movq	_xsrc@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _hxr00:
; DARWIN-64-DYNAMIC: 	movq	_xsrc@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _hxr00:
; DARWIN-64-PIC: 	movq	_xsrc@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @har01() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @dst to i8*)
; LINUX-64-STATIC-LABEL: har01:
; LINUX-64-STATIC: movl    $dst, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: har01:
; LINUX-32-STATIC: 	movl	$dst, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: har01:
; LINUX-32-PIC: 	movl	$dst, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: har01:
; LINUX-64-PIC: 	movq	dst@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _har01:
; DARWIN-32-STATIC: 	movl	$_dst, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _har01:
; DARWIN-32-DYNAMIC: 	movl	L_dst$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _har01:
; DARWIN-32-PIC: 	calll	L79$pb
; DARWIN-32-PIC-NEXT: L79$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L79$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _har01:
; DARWIN-64-STATIC: 	movq	_dst@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _har01:
; DARWIN-64-DYNAMIC: 	movq	_dst@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _har01:
; DARWIN-64-PIC: 	movq	_dst@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @hxr01() nounwind {
entry:
	ret i8* bitcast ([32 x i32]* @xdst to i8*)
; LINUX-64-STATIC-LABEL: hxr01:
; LINUX-64-STATIC: movl    $xdst, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: hxr01:
; LINUX-32-STATIC: 	movl	$xdst, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: hxr01:
; LINUX-32-PIC: 	movl	$xdst, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: hxr01:
; LINUX-64-PIC: 	movq	xdst@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _hxr01:
; DARWIN-32-STATIC: 	movl	$_xdst, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _hxr01:
; DARWIN-32-DYNAMIC: 	movl	L_xdst$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _hxr01:
; DARWIN-32-PIC: 	calll	L80$pb
; DARWIN-32-PIC-NEXT: L80$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xdst$non_lazy_ptr-L80$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _hxr01:
; DARWIN-64-STATIC: 	movq	_xdst@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _hxr01:
; DARWIN-64-DYNAMIC: 	movq	_xdst@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _hxr01:
; DARWIN-64-PIC: 	movq	_xdst@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @har02() nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = bitcast i32* %0 to i8*
	ret i8* %1
; LINUX-64-STATIC-LABEL: har02:
; LINUX-64-STATIC: movq    ptr(%rip), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: har02:
; LINUX-32-STATIC: 	movl	ptr, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: har02:
; LINUX-32-PIC: 	movl	ptr, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: har02:
; LINUX-64-PIC: 	movq	ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RAX]]), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _har02:
; DARWIN-32-STATIC: 	movl	_ptr, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _har02:
; DARWIN-32-DYNAMIC: 	movl	L_ptr$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[EAX]]), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _har02:
; DARWIN-32-PIC: 	calll	L81$pb
; DARWIN-32-PIC-NEXT: L81$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L81$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _har02:
; DARWIN-64-STATIC: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RAX]]), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _har02:
; DARWIN-64-DYNAMIC: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RAX]]), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _har02:
; DARWIN-64-PIC: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RAX]]), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @har03() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @dsrc to i8*)
; LINUX-64-STATIC-LABEL: har03:
; LINUX-64-STATIC: movl    $dsrc, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: har03:
; LINUX-32-STATIC: 	movl	$dsrc, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: har03:
; LINUX-32-PIC: 	movl	$dsrc, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: har03:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _har03:
; DARWIN-32-STATIC: 	movl	$_dsrc, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _har03:
; DARWIN-32-DYNAMIC: 	movl	$_dsrc, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _har03:
; DARWIN-32-PIC: 	calll	L82$pb
; DARWIN-32-PIC-NEXT: L82$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_dsrc-L82$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _har03:
; DARWIN-64-STATIC: 	leaq	_dsrc(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _har03:
; DARWIN-64-DYNAMIC: 	leaq	_dsrc(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _har03:
; DARWIN-64-PIC: 	leaq	_dsrc(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @har04() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @ddst to i8*)
; LINUX-64-STATIC-LABEL: har04:
; LINUX-64-STATIC: movl    $ddst, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: har04:
; LINUX-32-STATIC: 	movl	$ddst, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: har04:
; LINUX-32-PIC: 	movl	$ddst, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: har04:
; LINUX-64-PIC: 	movq	ddst@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _har04:
; DARWIN-32-STATIC: 	movl	$_ddst, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _har04:
; DARWIN-32-DYNAMIC: 	movl	$_ddst, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _har04:
; DARWIN-32-PIC: 	calll	L83$pb
; DARWIN-32-PIC-NEXT: L83$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_ddst-L83$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _har04:
; DARWIN-64-STATIC: 	leaq	_ddst(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _har04:
; DARWIN-64-DYNAMIC: 	leaq	_ddst(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _har04:
; DARWIN-64-PIC: 	leaq	_ddst(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @har05() nounwind {
entry:
	%0 = load i32*, i32** @dptr, align 8
	%1 = bitcast i32* %0 to i8*
	ret i8* %1
; LINUX-64-STATIC-LABEL: har05:
; LINUX-64-STATIC: movq    dptr(%rip), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: har05:
; LINUX-32-STATIC: 	movl	dptr, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: har05:
; LINUX-32-PIC: 	movl	dptr, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: har05:
; LINUX-64-PIC: 	movq	dptr@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RAX]]), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _har05:
; DARWIN-32-STATIC: 	movl	_dptr, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _har05:
; DARWIN-32-DYNAMIC: 	movl	_dptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _har05:
; DARWIN-32-PIC: 	calll	L84$pb
; DARWIN-32-PIC-NEXT: L84$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dptr-L84$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _har05:
; DARWIN-64-STATIC: 	movq	_dptr(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _har05:
; DARWIN-64-DYNAMIC: 	movq	_dptr(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _har05:
; DARWIN-64-PIC: 	movq	_dptr(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @har06() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @lsrc to i8*)
; LINUX-64-STATIC-LABEL: har06:
; LINUX-64-STATIC: movl    $lsrc, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: har06:
; LINUX-32-STATIC: 	movl	$lsrc, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: har06:
; LINUX-32-PIC: 	movl	$lsrc, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: har06:
; LINUX-64-PIC: 	leaq	lsrc(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _har06:
; DARWIN-32-STATIC: 	movl	$_lsrc, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _har06:
; DARWIN-32-DYNAMIC: 	movl	$_lsrc, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _har06:
; DARWIN-32-PIC: 	calll	L85$pb
; DARWIN-32-PIC-NEXT: L85$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_lsrc-L85$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _har06:
; DARWIN-64-STATIC: 	leaq	_lsrc(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _har06:
; DARWIN-64-DYNAMIC: 	leaq	_lsrc(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _har06:
; DARWIN-64-PIC: 	leaq	_lsrc(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @har07() nounwind {
entry:
	ret i8* bitcast ([131072 x i32]* @ldst to i8*)
; LINUX-64-STATIC-LABEL: har07:
; LINUX-64-STATIC: movl    $ldst, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: har07:
; LINUX-32-STATIC: 	movl	$ldst, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: har07:
; LINUX-32-PIC: 	movl	$ldst, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: har07:
; LINUX-64-PIC: 	leaq	ldst(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _har07:
; DARWIN-32-STATIC: 	movl	$_ldst, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _har07:
; DARWIN-32-DYNAMIC: 	movl	$_ldst, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _har07:
; DARWIN-32-PIC: 	calll	L86$pb
; DARWIN-32-PIC-NEXT: L86$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_ldst-L86$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _har07:
; DARWIN-64-STATIC: 	leaq	_ldst(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _har07:
; DARWIN-64-DYNAMIC: 	leaq	_ldst(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _har07:
; DARWIN-64-PIC: 	leaq	_ldst(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @har08() nounwind {
entry:
	%0 = load i32*, i32** @lptr, align 8
	%1 = bitcast i32* %0 to i8*
	ret i8* %1
; LINUX-64-STATIC-LABEL: har08:
; LINUX-64-STATIC: movq    lptr(%rip), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: har08:
; LINUX-32-STATIC: 	movl	lptr, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: har08:
; LINUX-32-PIC: 	movl	lptr, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: har08:
; LINUX-64-PIC: 	movq	lptr(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _har08:
; DARWIN-32-STATIC: 	movl	_lptr, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _har08:
; DARWIN-32-DYNAMIC: 	movl	_lptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _har08:
; DARWIN-32-PIC: 	calll	L87$pb
; DARWIN-32-PIC-NEXT: L87$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lptr-L87$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _har08:
; DARWIN-64-STATIC: 	movq	_lptr(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _har08:
; DARWIN-64-DYNAMIC: 	movq	_lptr(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _har08:
; DARWIN-64-PIC: 	movq	_lptr(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bat00() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32], [131072 x i32]* @src, i32 0, i64 16) to i8*)
; LINUX-64-STATIC-LABEL: bat00:
; LINUX-64-STATIC: movl    $src+64, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bat00:
; LINUX-32-STATIC: 	movl	$src+64, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bat00:
; LINUX-32-PIC: 	movl	$src+64, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bat00:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	addq	$64, %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bat00:
; DARWIN-32-STATIC: 	movl	$_src+64, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bat00:
; DARWIN-32-DYNAMIC: 	movl	L_src$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	addl	$64, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bat00:
; DARWIN-32-PIC: 	calll	L88$pb
; DARWIN-32-PIC-NEXT: L88$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L88$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	addl	$64, %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bat00:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	addq	$64, %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bat00:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	addq	$64, %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bat00:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	addq	$64, %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bxt00() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([32 x i32], [32 x i32]* @xsrc, i32 0, i64 16) to i8*)
; LINUX-64-STATIC-LABEL: bxt00:
; LINUX-64-STATIC: movl    $xsrc+64, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bxt00:
; LINUX-32-STATIC: 	movl	$xsrc+64, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bxt00:
; LINUX-32-PIC: 	movl	$xsrc+64, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bxt00:
; LINUX-64-PIC: 	movq	xsrc@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	addq	$64, %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bxt00:
; DARWIN-32-STATIC: 	movl	$_xsrc+64, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bxt00:
; DARWIN-32-DYNAMIC: 	movl	L_xsrc$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	addl	$64, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bxt00:
; DARWIN-32-PIC: 	calll	L89$pb
; DARWIN-32-PIC-NEXT: L89$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xsrc$non_lazy_ptr-L89$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	addl	$64, %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bxt00:
; DARWIN-64-STATIC: 	movq	_xsrc@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	addq	$64, %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bxt00:
; DARWIN-64-DYNAMIC: 	movq	_xsrc@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	addq	$64, %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bxt00:
; DARWIN-64-PIC: 	movq	_xsrc@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	addq	$64, %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bat01() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32], [131072 x i32]* @dst, i32 0, i64 16) to i8*)
; LINUX-64-STATIC-LABEL: bat01:
; LINUX-64-STATIC: movl    $dst+64, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bat01:
; LINUX-32-STATIC: 	movl	$dst+64, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bat01:
; LINUX-32-PIC: 	movl	$dst+64, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bat01:
; LINUX-64-PIC: 	movq	dst@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	addq	$64, %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bat01:
; DARWIN-32-STATIC: 	movl	$_dst+64, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bat01:
; DARWIN-32-DYNAMIC: 	movl	L_dst$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	addl	$64, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bat01:
; DARWIN-32-PIC: 	calll	L90$pb
; DARWIN-32-PIC-NEXT: L90$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L90$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	addl	$64, %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bat01:
; DARWIN-64-STATIC: 	movq	_dst@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	addq	$64, %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bat01:
; DARWIN-64-DYNAMIC: 	movq	_dst@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	addq	$64, %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bat01:
; DARWIN-64-PIC: 	movq	_dst@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	addq	$64, %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bxt01() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([32 x i32], [32 x i32]* @xdst, i32 0, i64 16) to i8*)
; LINUX-64-STATIC-LABEL: bxt01:
; LINUX-64-STATIC: movl    $xdst+64, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bxt01:
; LINUX-32-STATIC: 	movl	$xdst+64, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bxt01:
; LINUX-32-PIC: 	movl	$xdst+64, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bxt01:
; LINUX-64-PIC: 	movq	xdst@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	addq	$64, %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bxt01:
; DARWIN-32-STATIC: 	movl	$_xdst+64, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bxt01:
; DARWIN-32-DYNAMIC: 	movl	L_xdst$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	addl	$64, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bxt01:
; DARWIN-32-PIC: 	calll	L91$pb
; DARWIN-32-PIC-NEXT: L91$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xdst$non_lazy_ptr-L91$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	addl	$64, %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bxt01:
; DARWIN-64-STATIC: 	movq	_xdst@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	addq	$64, %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bxt01:
; DARWIN-64-DYNAMIC: 	movq	_xdst@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	addq	$64, %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bxt01:
; DARWIN-64-PIC: 	movq	_xdst@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	addq	$64, %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bat02() nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = getelementptr i32, i32* %0, i64 16
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: bat02:
; LINUX-64-STATIC: movq    ptr(%rip), %rax
; LINUX-64-STATIC: addq    $64, %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bat02:
; LINUX-32-STATIC: 	movl	ptr, %eax
; LINUX-32-STATIC-NEXT: 	addl	$64, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bat02:
; LINUX-32-PIC: 	movl	ptr, %eax
; LINUX-32-PIC-NEXT: 	addl	$64, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bat02:
; LINUX-64-PIC: 	movq	ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RAX]]), %rax
; LINUX-64-PIC-NEXT: 	addq	$64, %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bat02:
; DARWIN-32-STATIC: 	movl	_ptr, %eax
; DARWIN-32-STATIC-NEXT: 	addl	$64, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bat02:
; DARWIN-32-DYNAMIC: 	movl	L_ptr$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	addl	$64, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bat02:
; DARWIN-32-PIC: 	calll	L92$pb
; DARWIN-32-PIC-NEXT: L92$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L92$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	addl	$64, %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bat02:
; DARWIN-64-STATIC: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RAX]]), %rax
; DARWIN-64-STATIC-NEXT: 	addq	$64, %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bat02:
; DARWIN-64-DYNAMIC: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RAX]]), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	addq	$64, %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bat02:
; DARWIN-64-PIC: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RAX]]), %rax
; DARWIN-64-PIC-NEXT: 	addq	$64, %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bat03() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32], [131072 x i32]* @dsrc, i32 0, i64 16) to i8*)
; LINUX-64-STATIC-LABEL: bat03:
; LINUX-64-STATIC: movl    $dsrc+64, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bat03:
; LINUX-32-STATIC: 	movl	$dsrc+64, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bat03:
; LINUX-32-PIC: 	movl	$dsrc+64, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bat03:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	addq	$64, %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bat03:
; DARWIN-32-STATIC: 	movl	$_dsrc+64, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bat03:
; DARWIN-32-DYNAMIC: 	movl	$_dsrc+64, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bat03:
; DARWIN-32-PIC: 	calll	L93$pb
; DARWIN-32-PIC-NEXT: L93$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_dsrc-L93$pb)+64([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bat03:
; DARWIN-64-STATIC: 	leaq	_dsrc+64(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bat03:
; DARWIN-64-DYNAMIC: 	leaq	_dsrc+64(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bat03:
; DARWIN-64-PIC: 	leaq	_dsrc+64(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bat04() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32], [131072 x i32]* @ddst, i32 0, i64 16) to i8*)
; LINUX-64-STATIC-LABEL: bat04:
; LINUX-64-STATIC: movl    $ddst+64, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bat04:
; LINUX-32-STATIC: 	movl	$ddst+64, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bat04:
; LINUX-32-PIC: 	movl	$ddst+64, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bat04:
; LINUX-64-PIC: 	movq	ddst@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	addq	$64, %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bat04:
; DARWIN-32-STATIC: 	movl	$_ddst+64, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bat04:
; DARWIN-32-DYNAMIC: 	movl	$_ddst+64, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bat04:
; DARWIN-32-PIC: 	calll	L94$pb
; DARWIN-32-PIC-NEXT: L94$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ddst-L94$pb)+64([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bat04:
; DARWIN-64-STATIC: 	leaq	_ddst+64(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bat04:
; DARWIN-64-DYNAMIC: 	leaq	_ddst+64(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bat04:
; DARWIN-64-PIC: 	leaq	_ddst+64(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bat05() nounwind {
entry:
	%0 = load i32*, i32** @dptr, align 8
	%1 = getelementptr i32, i32* %0, i64 16
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: bat05:
; LINUX-64-STATIC: movq    dptr(%rip), %rax
; LINUX-64-STATIC: addq    $64, %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bat05:
; LINUX-32-STATIC: 	movl	dptr, %eax
; LINUX-32-STATIC-NEXT: 	addl	$64, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bat05:
; LINUX-32-PIC: 	movl	dptr, %eax
; LINUX-32-PIC-NEXT: 	addl	$64, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bat05:
; LINUX-64-PIC: 	movq	dptr@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RAX]]), %rax
; LINUX-64-PIC-NEXT: 	addq	$64, %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bat05:
; DARWIN-32-STATIC: 	movl	_dptr, %eax
; DARWIN-32-STATIC-NEXT: 	addl	$64, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bat05:
; DARWIN-32-DYNAMIC: 	movl	_dptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	addl	$64, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bat05:
; DARWIN-32-PIC: 	calll	L95$pb
; DARWIN-32-PIC-NEXT: L95$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dptr-L95$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	addl	$64, %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bat05:
; DARWIN-64-STATIC: 	movq	_dptr(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	addq	$64, %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bat05:
; DARWIN-64-DYNAMIC: 	movq	_dptr(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	addq	$64, %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bat05:
; DARWIN-64-PIC: 	movq	_dptr(%rip), %rax
; DARWIN-64-PIC-NEXT: 	addq	$64, %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bat06() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32], [131072 x i32]* @lsrc, i32 0, i64 16) to i8*)
; LINUX-64-STATIC-LABEL: bat06:
; LINUX-64-STATIC: movl    $lsrc+64, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bat06:
; LINUX-32-STATIC: 	movl	$lsrc+64, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bat06:
; LINUX-32-PIC: 	movl	$lsrc+64, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bat06:
; LINUX-64-PIC: 	leaq	lsrc+64(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bat06:
; DARWIN-32-STATIC: 	movl	$_lsrc+64, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bat06:
; DARWIN-32-DYNAMIC: 	movl	$_lsrc+64, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bat06:
; DARWIN-32-PIC: 	calll	L96$pb
; DARWIN-32-PIC-NEXT: L96$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_lsrc-L96$pb)+64([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bat06:
; DARWIN-64-STATIC: 	leaq	_lsrc+64(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bat06:
; DARWIN-64-DYNAMIC: 	leaq	_lsrc+64(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bat06:
; DARWIN-64-PIC: 	leaq	_lsrc+64(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bat07() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32], [131072 x i32]* @ldst, i32 0, i64 16) to i8*)
; LINUX-64-STATIC-LABEL: bat07:
; LINUX-64-STATIC: movl    $ldst+64, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bat07:
; LINUX-32-STATIC: 	movl	$ldst+64, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bat07:
; LINUX-32-PIC: 	movl	$ldst+64, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bat07:
; LINUX-64-PIC: 	leaq	ldst+64(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bat07:
; DARWIN-32-STATIC: 	movl	$_ldst+64, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bat07:
; DARWIN-32-DYNAMIC: 	movl	$_ldst+64, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bat07:
; DARWIN-32-PIC: 	calll	L97$pb
; DARWIN-32-PIC-NEXT: L97$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ldst-L97$pb)+64([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bat07:
; DARWIN-64-STATIC: 	leaq	_ldst+64(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bat07:
; DARWIN-64-DYNAMIC: 	leaq	_ldst+64(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bat07:
; DARWIN-64-PIC: 	leaq	_ldst+64(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bat08() nounwind {
entry:
	%0 = load i32*, i32** @lptr, align 8
	%1 = getelementptr i32, i32* %0, i64 16
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: bat08:
; LINUX-64-STATIC: movq    lptr(%rip), %rax
; LINUX-64-STATIC: addq    $64, %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bat08:
; LINUX-32-STATIC: 	movl	lptr, %eax
; LINUX-32-STATIC-NEXT: 	addl	$64, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bat08:
; LINUX-32-PIC: 	movl	lptr, %eax
; LINUX-32-PIC-NEXT: 	addl	$64, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bat08:
; LINUX-64-PIC: 	movq	lptr(%rip), %rax
; LINUX-64-PIC-NEXT: 	addq	$64, %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bat08:
; DARWIN-32-STATIC: 	movl	_lptr, %eax
; DARWIN-32-STATIC-NEXT: 	addl	$64, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bat08:
; DARWIN-32-DYNAMIC: 	movl	_lptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	addl	$64, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bat08:
; DARWIN-32-PIC: 	calll	L98$pb
; DARWIN-32-PIC-NEXT: L98$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lptr-L98$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	addl	$64, %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bat08:
; DARWIN-64-STATIC: 	movq	_lptr(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	addq	$64, %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bat08:
; DARWIN-64-DYNAMIC: 	movq	_lptr(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	addq	$64, %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bat08:
; DARWIN-64-PIC: 	movq	_lptr(%rip), %rax
; DARWIN-64-PIC-NEXT: 	addq	$64, %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bam00() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32], [131072 x i32]* @src, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC-LABEL: bam00:
; LINUX-64-STATIC: movl    $src+262144, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bam00:
; LINUX-32-STATIC: 	movl	$src+262144, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bam00:
; LINUX-32-PIC: 	movl	$src+262144, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bam00:
; LINUX-64-PIC: 	movl	$262144, %eax
; LINUX-64-PIC-NEXT: 	addq	src@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bam00:
; DARWIN-32-STATIC: 	movl	$_src+262144, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bam00:
; DARWIN-32-DYNAMIC: 	movl	$262144, %eax
; DARWIN-32-DYNAMIC-NEXT: 	addl	L_src$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bam00:
; DARWIN-32-PIC: 	calll	L99$pb
; DARWIN-32-PIC-NEXT: L99$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	$262144, %eax
; DARWIN-32-PIC-NEXT: 	addl	L_src$non_lazy_ptr-L99$pb([[ECX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bam00:
; DARWIN-64-STATIC: 	movl	$262144, %eax
; DARWIN-64-STATIC-NEXT: 	addq	_src@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bam00:
; DARWIN-64-DYNAMIC: 	movl	$262144, %eax
; DARWIN-64-DYNAMIC-NEXT: 	addq	_src@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bam00:
; DARWIN-64-PIC: 	movl	$262144, %eax
; DARWIN-64-PIC-NEXT: 	addq	_src@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bam01() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32], [131072 x i32]* @dst, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC-LABEL: bam01:
; LINUX-64-STATIC: movl    $dst+262144, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bam01:
; LINUX-32-STATIC: 	movl	$dst+262144, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bam01:
; LINUX-32-PIC: 	movl	$dst+262144, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bam01:
; LINUX-64-PIC: 	movl	$262144, %eax
; LINUX-64-PIC-NEXT: 	addq	dst@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bam01:
; DARWIN-32-STATIC: 	movl	$_dst+262144, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bam01:
; DARWIN-32-DYNAMIC: 	movl	$262144, %eax
; DARWIN-32-DYNAMIC-NEXT: 	addl	L_dst$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bam01:
; DARWIN-32-PIC: 	calll	L100$pb
; DARWIN-32-PIC-NEXT: L100$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	$262144, %eax
; DARWIN-32-PIC-NEXT: 	addl	L_dst$non_lazy_ptr-L100$pb([[ECX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bam01:
; DARWIN-64-STATIC: 	movl	$262144, %eax
; DARWIN-64-STATIC-NEXT: 	addq	_dst@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bam01:
; DARWIN-64-DYNAMIC: 	movl	$262144, %eax
; DARWIN-64-DYNAMIC-NEXT: 	addq	_dst@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bam01:
; DARWIN-64-PIC: 	movl	$262144, %eax
; DARWIN-64-PIC-NEXT: 	addq	_dst@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bxm01() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([32 x i32], [32 x i32]* @xdst, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC-LABEL: bxm01:
; LINUX-64-STATIC: movl    $xdst+262144, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bxm01:
; LINUX-32-STATIC: 	movl	$xdst+262144, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bxm01:
; LINUX-32-PIC: 	movl	$xdst+262144, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bxm01:
; LINUX-64-PIC: 	movl	$262144, %eax
; LINUX-64-PIC-NEXT: 	addq	xdst@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bxm01:
; DARWIN-32-STATIC: 	movl	$_xdst+262144, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bxm01:
; DARWIN-32-DYNAMIC: 	movl	$262144, %eax
; DARWIN-32-DYNAMIC-NEXT: 	addl	L_xdst$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bxm01:
; DARWIN-32-PIC: 	calll	L101$pb
; DARWIN-32-PIC-NEXT: L101$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	$262144, %eax
; DARWIN-32-PIC-NEXT: 	addl	L_xdst$non_lazy_ptr-L101$pb([[ECX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bxm01:
; DARWIN-64-STATIC: 	movl	$262144, %eax
; DARWIN-64-STATIC-NEXT: 	addq	_xdst@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bxm01:
; DARWIN-64-DYNAMIC: 	movl	$262144, %eax
; DARWIN-64-DYNAMIC-NEXT: 	addq	_xdst@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bxm01:
; DARWIN-64-PIC: 	movl	$262144, %eax
; DARWIN-64-PIC-NEXT: 	addq	_xdst@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bam02() nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = getelementptr i32, i32* %0, i64 65536
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: bam02:
; LINUX-64-STATIC: movl    $262144, %eax
; LINUX-64-STATIC: addq    ptr(%rip), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bam02:
; LINUX-32-STATIC: 	movl	$262144, %eax
; LINUX-32-STATIC-NEXT: 	addl	ptr, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bam02:
; LINUX-32-PIC: 	movl	$262144, %eax
; LINUX-32-PIC-NEXT: 	addl	ptr, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bam02:
; LINUX-64-PIC: 	movl	$262144, %eax
; LINUX-64-PIC-NEXT: 	movq	ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	addq	([[RCX]]), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bam02:
; DARWIN-32-STATIC: 	movl	$262144, %eax
; DARWIN-32-STATIC-NEXT: 	addl	_ptr, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bam02:
; DARWIN-32-DYNAMIC: 	movl	L_ptr$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	$262144, %eax
; DARWIN-32-DYNAMIC-NEXT: 	addl	([[ECX]]), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bam02:
; DARWIN-32-PIC: 	calll	L102$pb
; DARWIN-32-PIC-NEXT: L102$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L102$pb([[EAX]]), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	$262144, %eax
; DARWIN-32-PIC-NEXT: 	addl	([[ECX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bam02:
; DARWIN-64-STATIC: 	movl	$262144, %eax
; DARWIN-64-STATIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	addq	([[RCX]]), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bam02:
; DARWIN-64-DYNAMIC: 	movl	$262144, %eax
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	addq	([[RCX]]), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bam02:
; DARWIN-64-PIC: 	movl	$262144, %eax
; DARWIN-64-PIC-NEXT: 	movq	_ptr@GOTPCREL(%rip), [[RCX:%r.x]]
; DARWIN-64-PIC-NEXT: 	addq	([[RCX]]), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bam03() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32], [131072 x i32]* @dsrc, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC-LABEL: bam03:
; LINUX-64-STATIC: movl    $dsrc+262144, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bam03:
; LINUX-32-STATIC: 	movl	$dsrc+262144, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bam03:
; LINUX-32-PIC: 	movl	$dsrc+262144, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bam03:
; LINUX-64-PIC: 	movl	$262144, %eax
; LINUX-64-PIC-NEXT: 	addq	dsrc@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bam03:
; DARWIN-32-STATIC: 	movl	$_dsrc+262144, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bam03:
; DARWIN-32-DYNAMIC: 	movl	$_dsrc+262144, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bam03:
; DARWIN-32-PIC: 	calll	L103$pb
; DARWIN-32-PIC-NEXT: L103$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_dsrc-L103$pb)+262144([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bam03:
; DARWIN-64-STATIC: 	leaq	_dsrc+262144(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bam03:
; DARWIN-64-DYNAMIC: 	leaq	_dsrc+262144(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bam03:
; DARWIN-64-PIC: 	leaq	_dsrc+262144(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bam04() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32], [131072 x i32]* @ddst, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC-LABEL: bam04:
; LINUX-64-STATIC: movl    $ddst+262144, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bam04:
; LINUX-32-STATIC: 	movl	$ddst+262144, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bam04:
; LINUX-32-PIC: 	movl	$ddst+262144, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bam04:
; LINUX-64-PIC: 	movl	$262144, %eax
; LINUX-64-PIC-NEXT: 	addq	ddst@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bam04:
; DARWIN-32-STATIC: 	movl	$_ddst+262144, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bam04:
; DARWIN-32-DYNAMIC: 	movl	$_ddst+262144, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bam04:
; DARWIN-32-PIC: 	calll	L104$pb
; DARWIN-32-PIC-NEXT: L104$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ddst-L104$pb)+262144([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bam04:
; DARWIN-64-STATIC: 	leaq	_ddst+262144(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bam04:
; DARWIN-64-DYNAMIC: 	leaq	_ddst+262144(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bam04:
; DARWIN-64-PIC: 	leaq	_ddst+262144(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bam05() nounwind {
entry:
	%0 = load i32*, i32** @dptr, align 8
	%1 = getelementptr i32, i32* %0, i64 65536
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: bam05:
; LINUX-64-STATIC: movl    $262144, %eax
; LINUX-64-STATIC: addq    dptr(%rip), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bam05:
; LINUX-32-STATIC: 	movl	$262144, %eax
; LINUX-32-STATIC-NEXT: 	addl	dptr, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bam05:
; LINUX-32-PIC: 	movl	$262144, %eax
; LINUX-32-PIC-NEXT: 	addl	dptr, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bam05:
; LINUX-64-PIC: 	movl	$262144, %eax
; LINUX-64-PIC-NEXT: 	movq	dptr@GOTPCREL(%rip), [[RCX:%r.x]]
; LINUX-64-PIC-NEXT: 	addq	([[RCX]]), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bam05:
; DARWIN-32-STATIC: 	movl	$262144, %eax
; DARWIN-32-STATIC-NEXT: 	addl	_dptr, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bam05:
; DARWIN-32-DYNAMIC: 	movl	$262144, %eax
; DARWIN-32-DYNAMIC-NEXT: 	addl	_dptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bam05:
; DARWIN-32-PIC: 	calll	L105$pb
; DARWIN-32-PIC-NEXT: L105$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	$262144, %eax
; DARWIN-32-PIC-NEXT: 	addl	_dptr-L105$pb([[ECX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bam05:
; DARWIN-64-STATIC: 	movl	$262144, %eax
; DARWIN-64-STATIC-NEXT: 	addq	_dptr(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bam05:
; DARWIN-64-DYNAMIC: 	movl	$262144, %eax
; DARWIN-64-DYNAMIC-NEXT: 	addq	_dptr(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bam05:
; DARWIN-64-PIC: 	movl	$262144, %eax
; DARWIN-64-PIC-NEXT: 	addq	_dptr(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bam06() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32], [131072 x i32]* @lsrc, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC-LABEL: bam06:
; LINUX-64-STATIC: movl    $lsrc+262144, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bam06:
; LINUX-32-STATIC: 	movl	$lsrc+262144, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bam06:
; LINUX-32-PIC: 	movl	$lsrc+262144, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bam06:
; LINUX-64-PIC: 	leaq	lsrc+262144(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bam06:
; DARWIN-32-STATIC: 	movl	$_lsrc+262144, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bam06:
; DARWIN-32-DYNAMIC: 	movl	$_lsrc+262144, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bam06:
; DARWIN-32-PIC: 	calll	L106$pb
; DARWIN-32-PIC-NEXT: L106$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_lsrc-L106$pb)+262144([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bam06:
; DARWIN-64-STATIC: 	leaq	_lsrc+262144(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bam06:
; DARWIN-64-DYNAMIC: 	leaq	_lsrc+262144(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bam06:
; DARWIN-64-PIC: 	leaq	_lsrc+262144(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bam07() nounwind {
entry:
	ret i8* bitcast (i32* getelementptr ([131072 x i32], [131072 x i32]* @ldst, i32 0, i64 65536) to i8*)
; LINUX-64-STATIC-LABEL: bam07:
; LINUX-64-STATIC: movl    $ldst+262144, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bam07:
; LINUX-32-STATIC: 	movl	$ldst+262144, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bam07:
; LINUX-32-PIC: 	movl	$ldst+262144, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bam07:
; LINUX-64-PIC: 	leaq	ldst+262144(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bam07:
; DARWIN-32-STATIC: 	movl	$_ldst+262144, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bam07:
; DARWIN-32-DYNAMIC: 	movl	$_ldst+262144, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bam07:
; DARWIN-32-PIC: 	calll	L107$pb
; DARWIN-32-PIC-NEXT: L107$pb:
; DARWIN-32-PIC-NEXT: 	popl	%eax
; DARWIN-32-PIC-NEXT: 	leal	(_ldst-L107$pb)+262144([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bam07:
; DARWIN-64-STATIC: 	leaq	_ldst+262144(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bam07:
; DARWIN-64-DYNAMIC: 	leaq	_ldst+262144(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bam07:
; DARWIN-64-PIC: 	leaq	_ldst+262144(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @bam08() nounwind {
entry:
	%0 = load i32*, i32** @lptr, align 8
	%1 = getelementptr i32, i32* %0, i64 65536
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: bam08:
; LINUX-64-STATIC: movl    $262144, %eax
; LINUX-64-STATIC: addq    lptr(%rip), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: bam08:
; LINUX-32-STATIC: 	movl	$262144, %eax
; LINUX-32-STATIC-NEXT: 	addl	lptr, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: bam08:
; LINUX-32-PIC: 	movl	$262144, %eax
; LINUX-32-PIC-NEXT: 	addl	lptr, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: bam08:
; LINUX-64-PIC: 	movl	$262144, %eax
; LINUX-64-PIC-NEXT: 	addq	lptr(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _bam08:
; DARWIN-32-STATIC: 	movl	$262144, %eax
; DARWIN-32-STATIC-NEXT: 	addl	_lptr, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _bam08:
; DARWIN-32-DYNAMIC: 	movl	$262144, %eax
; DARWIN-32-DYNAMIC-NEXT: 	addl	_lptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _bam08:
; DARWIN-32-PIC: 	calll	L108$pb
; DARWIN-32-PIC-NEXT: L108$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	$262144, %eax
; DARWIN-32-PIC-NEXT: 	addl	_lptr-L108$pb([[ECX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _bam08:
; DARWIN-64-STATIC: 	movl	$262144, %eax
; DARWIN-64-STATIC-NEXT: 	addq	_lptr(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _bam08:
; DARWIN-64-DYNAMIC: 	movl	$262144, %eax
; DARWIN-64-DYNAMIC-NEXT: 	addq	_lptr(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _bam08:
; DARWIN-64-PIC: 	movl	$262144, %eax
; DARWIN-64-PIC-NEXT: 	addq	_lptr(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cat00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32], [131072 x i32]* @src, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cat00:
; LINUX-64-STATIC: leaq    src+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cat00:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	src+64(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cat00:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	src+64(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cat00:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cat00:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_src+64(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cat00:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_src$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cat00:
; DARWIN-32-PIC: 	calll	L109$pb
; DARWIN-32-PIC-NEXT: L109$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L109$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	64([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cat00:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cat00:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cat00:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cxt00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [32 x i32], [32 x i32]* @xsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cxt00:
; LINUX-64-STATIC: leaq    xsrc+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cxt00:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	xsrc+64(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cxt00:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	xsrc+64(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cxt00:
; LINUX-64-PIC: 	movq	xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cxt00:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_xsrc+64(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cxt00:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_xsrc$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cxt00:
; DARWIN-32-PIC: 	calll	L110$pb
; DARWIN-32-PIC-NEXT: L110$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xsrc$non_lazy_ptr-L110$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	64([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cxt00:
; DARWIN-64-STATIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cxt00:
; DARWIN-64-DYNAMIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cxt00:
; DARWIN-64-PIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cat01(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32], [131072 x i32]* @dst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cat01:
; LINUX-64-STATIC: leaq    dst+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cat01:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	dst+64(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cat01:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	dst+64(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cat01:
; LINUX-64-PIC: 	movq	dst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cat01:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_dst+64(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cat01:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_dst$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cat01:
; DARWIN-32-PIC: 	calll	L111$pb
; DARWIN-32-PIC-NEXT: L111$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L111$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	64([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cat01:
; DARWIN-64-STATIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cat01:
; DARWIN-64-DYNAMIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cat01:
; DARWIN-64-PIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cxt01(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [32 x i32], [32 x i32]* @xdst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cxt01:
; LINUX-64-STATIC: leaq    xdst+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cxt01:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	xdst+64(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cxt01:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	xdst+64(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cxt01:
; LINUX-64-PIC: 	movq	xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cxt01:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_xdst+64(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cxt01:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_xdst$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cxt01:
; DARWIN-32-PIC: 	calll	L112$pb
; DARWIN-32-PIC-NEXT: L112$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xdst$non_lazy_ptr-L112$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	64([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cxt01:
; DARWIN-64-STATIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cxt01:
; DARWIN-64-DYNAMIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cxt01:
; DARWIN-64-PIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cat02(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr i32, i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
; LINUX-64-STATIC-LABEL: cat02:
; LINUX-64-STATIC: movq    ptr(%rip), [[RAX:%r.x]]
; LINUX-64-STATIC: leaq    64([[RAX]],%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cat02:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cat02:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cat02:
; LINUX-64-PIC: 	movq	ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RAX]]), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cat02:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_ptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cat02:
; DARWIN-32-DYNAMIC: 	movl	L_ptr$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	64([[EAX]],[[ECX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cat02:
; DARWIN-32-PIC: 	calll	L113$pb
; DARWIN-32-PIC-NEXT: L113$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L113$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	64([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cat02:
; DARWIN-64-STATIC: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RAX]]), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cat02:
; DARWIN-64-DYNAMIC: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RAX]]), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cat02:
; DARWIN-64-PIC: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RAX]]), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cat03(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32], [131072 x i32]* @dsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cat03:
; LINUX-64-STATIC: leaq    dsrc+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cat03:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	dsrc+64(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cat03:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	dsrc+64(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cat03:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cat03:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_dsrc+64(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cat03:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_dsrc+64(,[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cat03:
; DARWIN-32-PIC: 	calll	L114$pb
; DARWIN-32-PIC-NEXT: L114$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_dsrc-L114$pb)+64([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cat03:
; DARWIN-64-STATIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cat03:
; DARWIN-64-DYNAMIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cat03:
; DARWIN-64-PIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cat04(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32], [131072 x i32]* @ddst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cat04:
; LINUX-64-STATIC: leaq    ddst+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cat04:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	ddst+64(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cat04:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	ddst+64(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cat04:
; LINUX-64-PIC: 	movq	ddst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cat04:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_ddst+64(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cat04:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_ddst+64(,[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cat04:
; DARWIN-32-PIC: 	calll	L115$pb
; DARWIN-32-PIC-NEXT: L115$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ddst-L115$pb)+64([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cat04:
; DARWIN-64-STATIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cat04:
; DARWIN-64-DYNAMIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cat04:
; DARWIN-64-PIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cat05(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @dptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr i32, i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
; LINUX-64-STATIC-LABEL: cat05:
; LINUX-64-STATIC: movq    dptr(%rip), [[RAX:%r.x]]
; LINUX-64-STATIC: leaq    64([[RAX]],%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cat05:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cat05:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cat05:
; LINUX-64-PIC: 	movq	dptr@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RAX]]), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cat05:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cat05:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cat05:
; DARWIN-32-PIC: 	calll	L116$pb
; DARWIN-32-PIC-NEXT: L116$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dptr-L116$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	64([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cat05:
; DARWIN-64-STATIC: 	movq	_dptr(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cat05:
; DARWIN-64-DYNAMIC: 	movq	_dptr(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cat05:
; DARWIN-64-PIC: 	movq	_dptr(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cat06(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32], [131072 x i32]* @lsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cat06:
; LINUX-64-STATIC: leaq    lsrc+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cat06:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	lsrc+64(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cat06:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	lsrc+64(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cat06:
; LINUX-64-PIC: 	leaq	lsrc(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cat06:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_lsrc+64(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cat06:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_lsrc+64(,[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cat06:
; DARWIN-32-PIC: 	calll	L117$pb
; DARWIN-32-PIC-NEXT: L117$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_lsrc-L117$pb)+64([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cat06:
; DARWIN-64-STATIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cat06:
; DARWIN-64-DYNAMIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cat06:
; DARWIN-64-PIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cat07(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 16
	%1 = getelementptr [131072 x i32], [131072 x i32]* @ldst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cat07:
; LINUX-64-STATIC: leaq    ldst+64(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cat07:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	ldst+64(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cat07:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	ldst+64(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cat07:
; LINUX-64-PIC: 	leaq	ldst(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cat07:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_ldst+64(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cat07:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_ldst+64(,[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cat07:
; DARWIN-32-PIC: 	calll	L118$pb
; DARWIN-32-PIC-NEXT: L118$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ldst-L118$pb)+64([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cat07:
; DARWIN-64-STATIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cat07:
; DARWIN-64-DYNAMIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cat07:
; DARWIN-64-PIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cat08(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @lptr, align 8
	%1 = add i64 %i, 16
	%2 = getelementptr i32, i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
; LINUX-64-STATIC-LABEL: cat08:
; LINUX-64-STATIC: movq    lptr(%rip), [[RAX:%r.x]]
; LINUX-64-STATIC: leaq    64([[RAX]],%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cat08:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cat08:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cat08:
; LINUX-64-PIC: 	movq	lptr(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cat08:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cat08:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	64([[ECX]],[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cat08:
; DARWIN-32-PIC: 	calll	L119$pb
; DARWIN-32-PIC-NEXT: L119$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lptr-L119$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	64([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cat08:
; DARWIN-64-STATIC: 	movq	_lptr(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cat08:
; DARWIN-64-DYNAMIC: 	movq	_lptr(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cat08:
; DARWIN-64-PIC: 	movq	_lptr(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	64([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cam00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32], [131072 x i32]* @src, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cam00:
; LINUX-64-STATIC: leaq    src+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cam00:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	src+262144(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cam00:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	src+262144(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cam00:
; LINUX-64-PIC: 	movq	src@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cam00:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_src+262144(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cam00:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_src$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cam00:
; DARWIN-32-PIC: 	calll	L120$pb
; DARWIN-32-PIC-NEXT: L120$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_src$non_lazy_ptr-L120$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	262144([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cam00:
; DARWIN-64-STATIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cam00:
; DARWIN-64-DYNAMIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cam00:
; DARWIN-64-PIC: 	movq	_src@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cxm00(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [32 x i32], [32 x i32]* @xsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cxm00:
; LINUX-64-STATIC: leaq    xsrc+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cxm00:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	xsrc+262144(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cxm00:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	xsrc+262144(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cxm00:
; LINUX-64-PIC: 	movq	xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cxm00:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_xsrc+262144(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cxm00:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_xsrc$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cxm00:
; DARWIN-32-PIC: 	calll	L121$pb
; DARWIN-32-PIC-NEXT: L121$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xsrc$non_lazy_ptr-L121$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	262144([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cxm00:
; DARWIN-64-STATIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cxm00:
; DARWIN-64-DYNAMIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cxm00:
; DARWIN-64-PIC: 	movq	_xsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cam01(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32], [131072 x i32]* @dst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cam01:
; LINUX-64-STATIC: leaq    dst+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cam01:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	dst+262144(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cam01:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	dst+262144(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cam01:
; LINUX-64-PIC: 	movq	dst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cam01:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_dst+262144(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cam01:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_dst$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cam01:
; DARWIN-32-PIC: 	calll	L122$pb
; DARWIN-32-PIC-NEXT: L122$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_dst$non_lazy_ptr-L122$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	262144([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cam01:
; DARWIN-64-STATIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cam01:
; DARWIN-64-DYNAMIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cam01:
; DARWIN-64-PIC: 	movq	_dst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cxm01(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [32 x i32], [32 x i32]* @xdst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cxm01:
; LINUX-64-STATIC: leaq    xdst+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cxm01:
; LINUX-32-STATIC: 	movl	4(%esp), %eax
; LINUX-32-STATIC-NEXT: 	leal	xdst+262144(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cxm01:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	xdst+262144(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cxm01:
; LINUX-64-PIC: 	movq	xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cxm01:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_xdst+262144(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cxm01:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_xdst$non_lazy_ptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cxm01:
; DARWIN-32-PIC: 	calll	L123$pb
; DARWIN-32-PIC-NEXT: L123$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_xdst$non_lazy_ptr-L123$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	262144([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cxm01:
; DARWIN-64-STATIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cxm01:
; DARWIN-64-DYNAMIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cxm01:
; DARWIN-64-PIC: 	movq	_xdst@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cam02(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @ptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr i32, i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
; LINUX-64-STATIC-LABEL: cam02:
; LINUX-64-STATIC: movq    ptr(%rip), [[RAX:%r.x]]
; LINUX-64-STATIC: leaq    262144([[RAX]],%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cam02:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cam02:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	ptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cam02:
; LINUX-64-PIC: 	movq	ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RAX]]), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cam02:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_ptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cam02:
; DARWIN-32-DYNAMIC: 	movl	L_ptr$non_lazy_ptr, [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	262144([[EAX]],[[ECX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cam02:
; DARWIN-32-PIC: 	calll	L124$pb
; DARWIN-32-PIC-NEXT: L124$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ptr$non_lazy_ptr-L124$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	262144([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cam02:
; DARWIN-64-STATIC: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	([[RAX]]), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cam02:
; DARWIN-64-DYNAMIC: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	([[RAX]]), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cam02:
; DARWIN-64-PIC: 	movq	_ptr@GOTPCREL(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	([[RAX]]), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cam03(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32], [131072 x i32]* @dsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cam03:
; LINUX-64-STATIC: leaq    dsrc+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cam03:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	dsrc+262144(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cam03:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	dsrc+262144(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cam03:
; LINUX-64-PIC: 	movq	dsrc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cam03:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_dsrc+262144(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cam03:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_dsrc+262144(,[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cam03:
; DARWIN-32-PIC: 	calll	L125$pb
; DARWIN-32-PIC-NEXT: L125$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_dsrc-L125$pb)+262144([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cam03:
; DARWIN-64-STATIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cam03:
; DARWIN-64-DYNAMIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cam03:
; DARWIN-64-PIC: 	leaq	_dsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cam04(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32], [131072 x i32]* @ddst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cam04:
; LINUX-64-STATIC: leaq    ddst+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cam04:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	ddst+262144(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cam04:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	ddst+262144(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cam04:
; LINUX-64-PIC: 	movq	ddst@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cam04:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_ddst+262144(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cam04:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_ddst+262144(,[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cam04:
; DARWIN-32-PIC: 	calll	L126$pb
; DARWIN-32-PIC-NEXT: L126$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ddst-L126$pb)+262144([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cam04:
; DARWIN-64-STATIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cam04:
; DARWIN-64-DYNAMIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cam04:
; DARWIN-64-PIC: 	leaq	_ddst(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cam05(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @dptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr i32, i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
; LINUX-64-STATIC-LABEL: cam05:
; LINUX-64-STATIC: movq    dptr(%rip), [[RAX:%r.x]]
; LINUX-64-STATIC: leaq    262144([[RAX]],%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cam05:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	dptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cam05:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	dptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cam05:
; LINUX-64-PIC: 	movq	dptr@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	([[RAX]]), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cam05:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_dptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cam05:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_dptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cam05:
; DARWIN-32-PIC: 	calll	L127$pb
; DARWIN-32-PIC-NEXT: L127$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_dptr-L127$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	262144([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cam05:
; DARWIN-64-STATIC: 	movq	_dptr(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cam05:
; DARWIN-64-DYNAMIC: 	movq	_dptr(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cam05:
; DARWIN-64-PIC: 	movq	_dptr(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cam06(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32], [131072 x i32]* @lsrc, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cam06:
; LINUX-64-STATIC: leaq    lsrc+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cam06:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	lsrc+262144(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cam06:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	lsrc+262144(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cam06:
; LINUX-64-PIC: 	leaq	lsrc(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cam06:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_lsrc+262144(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cam06:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_lsrc+262144(,[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cam06:
; DARWIN-32-PIC: 	calll	L128$pb
; DARWIN-32-PIC-NEXT: L128$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_lsrc-L128$pb)+262144([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cam06:
; DARWIN-64-STATIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cam06:
; DARWIN-64-DYNAMIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cam06:
; DARWIN-64-PIC: 	leaq	_lsrc(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cam07(i64 %i) nounwind {
entry:
	%0 = add i64 %i, 65536
	%1 = getelementptr [131072 x i32], [131072 x i32]* @ldst, i64 0, i64 %0
	%2 = bitcast i32* %1 to i8*
	ret i8* %2
; LINUX-64-STATIC-LABEL: cam07:
; LINUX-64-STATIC: leaq    ldst+262144(,%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cam07:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	ldst+262144(,[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cam07:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	ldst+262144(,[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cam07:
; LINUX-64-PIC: 	leaq	ldst(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cam07:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	_ldst+262144(,[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cam07:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	_ldst+262144(,[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cam07:
; DARWIN-32-PIC: 	calll	L129$pb
; DARWIN-32-PIC-NEXT: L129$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	(_ldst-L129$pb)+262144([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cam07:
; DARWIN-64-STATIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cam07:
; DARWIN-64-DYNAMIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cam07:
; DARWIN-64-PIC: 	leaq	_ldst(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define i8* @cam08(i64 %i) nounwind {
entry:
	%0 = load i32*, i32** @lptr, align 8
	%1 = add i64 %i, 65536
	%2 = getelementptr i32, i32* %0, i64 %1
	%3 = bitcast i32* %2 to i8*
	ret i8* %3
; LINUX-64-STATIC-LABEL: cam08:
; LINUX-64-STATIC: movq    lptr(%rip), [[RAX:%r.x]]
; LINUX-64-STATIC: leaq    262144([[RAX]],%rdi,4), %rax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: cam08:
; LINUX-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-STATIC-NEXT: 	movl	lptr, [[ECX:%e.x]]
; LINUX-32-STATIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: cam08:
; LINUX-32-PIC: 	movl	4(%esp), [[EAX:%e.x]]
; LINUX-32-PIC-NEXT: 	movl	lptr, [[ECX:%e.x]]
; LINUX-32-PIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: cam08:
; LINUX-64-PIC: 	movq	lptr(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _cam08:
; DARWIN-32-STATIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	movl	_lptr, [[ECX:%e.x]]
; DARWIN-32-STATIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _cam08:
; DARWIN-32-DYNAMIC: 	movl	4(%esp), [[EAX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	movl	_lptr, [[ECX:%e.x]]
; DARWIN-32-DYNAMIC-NEXT: 	leal	262144([[ECX]],[[EAX]],4), %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _cam08:
; DARWIN-32-PIC: 	calll	L130$pb
; DARWIN-32-PIC-NEXT: L130$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	4(%esp), [[ECX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	_lptr-L130$pb([[EAX]]), [[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	262144([[EAX]],[[ECX]],4), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _cam08:
; DARWIN-64-STATIC: 	movq	_lptr(%rip), [[RAX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _cam08:
; DARWIN-64-DYNAMIC: 	movq	_lptr(%rip), [[RAX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _cam08:
; DARWIN-64-PIC: 	movq	_lptr(%rip), [[RAX:%r.x]]
; DARWIN-64-PIC-NEXT: 	leaq	262144([[RAX]],%rdi,4), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define void @lcallee() nounwind {
entry:
	call void @x() nounwind
	call void @x() nounwind
	call void @x() nounwind
	call void @x() nounwind
	call void @x() nounwind
	call void @x() nounwind
	call void @x() nounwind
	ret void
; LINUX-64-STATIC-LABEL: lcallee:
; LINUX-64-STATIC: callq   x
; LINUX-64-STATIC: callq   x
; LINUX-64-STATIC: callq   x
; LINUX-64-STATIC: callq   x
; LINUX-64-STATIC: callq   x
; LINUX-64-STATIC: callq   x
; LINUX-64-STATIC: callq   x
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: lcallee:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	x
; LINUX-32-STATIC-NEXT: 	calll	x
; LINUX-32-STATIC-NEXT: 	calll	x
; LINUX-32-STATIC-NEXT: 	calll	x
; LINUX-32-STATIC-NEXT: 	calll	x
; LINUX-32-STATIC-NEXT: 	calll	x
; LINUX-32-STATIC-NEXT: 	calll	x
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: lcallee:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	x
; LINUX-32-PIC-NEXT: 	calll	x
; LINUX-32-PIC-NEXT: 	calll	x
; LINUX-32-PIC-NEXT: 	calll	x
; LINUX-32-PIC-NEXT: 	calll	x
; LINUX-32-PIC-NEXT: 	calll	x
; LINUX-32-PIC-NEXT: 	calll	x
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: lcallee:
; LINUX-64-PIC: 	pushq
; LINUX-64-PIC-NEXT: 	callq	x@PLT
; LINUX-64-PIC-NEXT: 	callq	x@PLT
; LINUX-64-PIC-NEXT: 	callq	x@PLT
; LINUX-64-PIC-NEXT: 	callq	x@PLT
; LINUX-64-PIC-NEXT: 	callq	x@PLT
; LINUX-64-PIC-NEXT: 	callq	x@PLT
; LINUX-64-PIC-NEXT: 	callq	x@PLT
; LINUX-64-PIC-NEXT: 	popq
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _lcallee:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	_x
; DARWIN-32-STATIC-NEXT: 	calll	_x
; DARWIN-32-STATIC-NEXT: 	calll	_x
; DARWIN-32-STATIC-NEXT: 	calll	_x
; DARWIN-32-STATIC-NEXT: 	calll	_x
; DARWIN-32-STATIC-NEXT: 	calll	_x
; DARWIN-32-STATIC-NEXT: 	calll	_x
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _lcallee:
; DARWIN-32-DYNAMIC: 	subl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	calll	_x
; DARWIN-32-DYNAMIC-NEXT: 	calll	_x
; DARWIN-32-DYNAMIC-NEXT: 	calll	_x
; DARWIN-32-DYNAMIC-NEXT: 	calll	_x
; DARWIN-32-DYNAMIC-NEXT: 	calll	_x
; DARWIN-32-DYNAMIC-NEXT: 	calll	_x
; DARWIN-32-DYNAMIC-NEXT: 	calll	_x
; DARWIN-32-DYNAMIC-NEXT: 	addl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _lcallee:
; DARWIN-32-PIC: 	subl	$12, %esp
; DARWIN-32-PIC-NEXT: 	calll	_x
; DARWIN-32-PIC-NEXT: 	calll	_x
; DARWIN-32-PIC-NEXT: 	calll	_x
; DARWIN-32-PIC-NEXT: 	calll	_x
; DARWIN-32-PIC-NEXT: 	calll	_x
; DARWIN-32-PIC-NEXT: 	calll	_x
; DARWIN-32-PIC-NEXT: 	calll	_x
; DARWIN-32-PIC-NEXT: 	addl	$12, %esp
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _lcallee:
; DARWIN-64-STATIC: 	pushq
; DARWIN-64-STATIC-NEXT: 	callq	_x
; DARWIN-64-STATIC-NEXT: 	callq	_x
; DARWIN-64-STATIC-NEXT: 	callq	_x
; DARWIN-64-STATIC-NEXT: 	callq	_x
; DARWIN-64-STATIC-NEXT: 	callq	_x
; DARWIN-64-STATIC-NEXT: 	callq	_x
; DARWIN-64-STATIC-NEXT: 	callq	_x
; DARWIN-64-STATIC-NEXT: 	popq
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _lcallee:
; DARWIN-64-DYNAMIC: 	pushq
; DARWIN-64-DYNAMIC-NEXT: 	callq	_x
; DARWIN-64-DYNAMIC-NEXT: 	callq	_x
; DARWIN-64-DYNAMIC-NEXT: 	callq	_x
; DARWIN-64-DYNAMIC-NEXT: 	callq	_x
; DARWIN-64-DYNAMIC-NEXT: 	callq	_x
; DARWIN-64-DYNAMIC-NEXT: 	callq	_x
; DARWIN-64-DYNAMIC-NEXT: 	callq	_x
; DARWIN-64-DYNAMIC-NEXT: 	popq
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _lcallee:
; DARWIN-64-PIC: 	pushq
; DARWIN-64-PIC-NEXT: 	callq	_x
; DARWIN-64-PIC-NEXT: 	callq	_x
; DARWIN-64-PIC-NEXT: 	callq	_x
; DARWIN-64-PIC-NEXT: 	callq	_x
; DARWIN-64-PIC-NEXT: 	callq	_x
; DARWIN-64-PIC-NEXT: 	callq	_x
; DARWIN-64-PIC-NEXT: 	callq	_x
; DARWIN-64-PIC-NEXT: 	popq
; DARWIN-64-PIC-NEXT: 	ret
}

declare void @x()

define internal void @dcallee() nounwind {
entry:
	call void @y() nounwind
	call void @y() nounwind
	call void @y() nounwind
	call void @y() nounwind
	call void @y() nounwind
	call void @y() nounwind
	call void @y() nounwind
	ret void
; LINUX-64-STATIC-LABEL: dcallee:
; LINUX-64-STATIC: callq   y
; LINUX-64-STATIC: callq   y
; LINUX-64-STATIC: callq   y
; LINUX-64-STATIC: callq   y
; LINUX-64-STATIC: callq   y
; LINUX-64-STATIC: callq   y
; LINUX-64-STATIC: callq   y
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: dcallee:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	y
; LINUX-32-STATIC-NEXT: 	calll	y
; LINUX-32-STATIC-NEXT: 	calll	y
; LINUX-32-STATIC-NEXT: 	calll	y
; LINUX-32-STATIC-NEXT: 	calll	y
; LINUX-32-STATIC-NEXT: 	calll	y
; LINUX-32-STATIC-NEXT: 	calll	y
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: dcallee:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	y
; LINUX-32-PIC-NEXT: 	calll	y
; LINUX-32-PIC-NEXT: 	calll	y
; LINUX-32-PIC-NEXT: 	calll	y
; LINUX-32-PIC-NEXT: 	calll	y
; LINUX-32-PIC-NEXT: 	calll	y
; LINUX-32-PIC-NEXT: 	calll	y
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: dcallee:
; LINUX-64-PIC: 	pushq
; LINUX-64-PIC-NEXT: 	callq	y@PLT
; LINUX-64-PIC-NEXT: 	callq	y@PLT
; LINUX-64-PIC-NEXT: 	callq	y@PLT
; LINUX-64-PIC-NEXT: 	callq	y@PLT
; LINUX-64-PIC-NEXT: 	callq	y@PLT
; LINUX-64-PIC-NEXT: 	callq	y@PLT
; LINUX-64-PIC-NEXT: 	callq	y@PLT
; LINUX-64-PIC-NEXT: 	popq
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _dcallee:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	_y
; DARWIN-32-STATIC-NEXT: 	calll	_y
; DARWIN-32-STATIC-NEXT: 	calll	_y
; DARWIN-32-STATIC-NEXT: 	calll	_y
; DARWIN-32-STATIC-NEXT: 	calll	_y
; DARWIN-32-STATIC-NEXT: 	calll	_y
; DARWIN-32-STATIC-NEXT: 	calll	_y
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _dcallee:
; DARWIN-32-DYNAMIC: 	subl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	calll	_y
; DARWIN-32-DYNAMIC-NEXT: 	calll	_y
; DARWIN-32-DYNAMIC-NEXT: 	calll	_y
; DARWIN-32-DYNAMIC-NEXT: 	calll	_y
; DARWIN-32-DYNAMIC-NEXT: 	calll	_y
; DARWIN-32-DYNAMIC-NEXT: 	calll	_y
; DARWIN-32-DYNAMIC-NEXT: 	calll	_y
; DARWIN-32-DYNAMIC-NEXT: 	addl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _dcallee:
; DARWIN-32-PIC: 	subl	$12, %esp
; DARWIN-32-PIC-NEXT: 	calll	_y
; DARWIN-32-PIC-NEXT: 	calll	_y
; DARWIN-32-PIC-NEXT: 	calll	_y
; DARWIN-32-PIC-NEXT: 	calll	_y
; DARWIN-32-PIC-NEXT: 	calll	_y
; DARWIN-32-PIC-NEXT: 	calll	_y
; DARWIN-32-PIC-NEXT: 	calll	_y
; DARWIN-32-PIC-NEXT: 	addl	$12, %esp
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _dcallee:
; DARWIN-64-STATIC: 	pushq
; DARWIN-64-STATIC-NEXT: 	callq	_y
; DARWIN-64-STATIC-NEXT: 	callq	_y
; DARWIN-64-STATIC-NEXT: 	callq	_y
; DARWIN-64-STATIC-NEXT: 	callq	_y
; DARWIN-64-STATIC-NEXT: 	callq	_y
; DARWIN-64-STATIC-NEXT: 	callq	_y
; DARWIN-64-STATIC-NEXT: 	callq	_y
; DARWIN-64-STATIC-NEXT: 	popq
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _dcallee:
; DARWIN-64-DYNAMIC: 	pushq
; DARWIN-64-DYNAMIC-NEXT: 	callq	_y
; DARWIN-64-DYNAMIC-NEXT: 	callq	_y
; DARWIN-64-DYNAMIC-NEXT: 	callq	_y
; DARWIN-64-DYNAMIC-NEXT: 	callq	_y
; DARWIN-64-DYNAMIC-NEXT: 	callq	_y
; DARWIN-64-DYNAMIC-NEXT: 	callq	_y
; DARWIN-64-DYNAMIC-NEXT: 	callq	_y
; DARWIN-64-DYNAMIC-NEXT: 	popq
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _dcallee:
; DARWIN-64-PIC: 	pushq
; DARWIN-64-PIC-NEXT: 	callq	_y
; DARWIN-64-PIC-NEXT: 	callq	_y
; DARWIN-64-PIC-NEXT: 	callq	_y
; DARWIN-64-PIC-NEXT: 	callq	_y
; DARWIN-64-PIC-NEXT: 	callq	_y
; DARWIN-64-PIC-NEXT: 	callq	_y
; DARWIN-64-PIC-NEXT: 	callq	_y
; DARWIN-64-PIC-NEXT: 	popq
; DARWIN-64-PIC-NEXT: 	ret
}

declare void @y()

define void ()* @address() nounwind {
entry:
	ret void ()* @callee
; LINUX-64-STATIC-LABEL: address:
; LINUX-64-STATIC: movl    $callee, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: address:
; LINUX-32-STATIC: 	movl	$callee, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: address:
; LINUX-32-PIC: 	movl	$callee, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: address:
; LINUX-64-PIC: 	movq	callee@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _address:
; DARWIN-32-STATIC: 	movl	$_callee, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _address:
; DARWIN-32-DYNAMIC: 	movl	L_callee$non_lazy_ptr, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _address:
; DARWIN-32-PIC: 	calll	L133$pb
; DARWIN-32-PIC-NEXT: L133$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_callee$non_lazy_ptr-L133$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _address:
; DARWIN-64-STATIC: 	movq	_callee@GOTPCREL(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _address:
; DARWIN-64-DYNAMIC: 	movq	_callee@GOTPCREL(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _address:
; DARWIN-64-PIC: 	movq	_callee@GOTPCREL(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

declare void @callee()

define void ()* @laddress() nounwind {
entry:
	ret void ()* @lcallee
; LINUX-64-STATIC-LABEL: laddress:
; LINUX-64-STATIC: movl    $lcallee, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: laddress:
; LINUX-32-STATIC: 	movl	$lcallee, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: laddress:
; LINUX-32-PIC: 	movl	$lcallee, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: laddress:
; LINUX-64-PIC: 	movq	lcallee@GOTPCREL(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _laddress:
; DARWIN-32-STATIC: 	movl	$_lcallee, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _laddress:
; DARWIN-32-DYNAMIC: 	movl	$_lcallee, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _laddress:
; DARWIN-32-PIC: 	calll	L134$pb
; DARWIN-32-PIC-NEXT: L134$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_lcallee-L134$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _laddress:
; DARWIN-64-STATIC: 	leaq	_lcallee(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _laddress:
; DARWIN-64-DYNAMIC: 	leaq	_lcallee(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _laddress:
; DARWIN-64-PIC: 	leaq	_lcallee(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define void ()* @daddress() nounwind {
entry:
	ret void ()* @dcallee
; LINUX-64-STATIC-LABEL: daddress:
; LINUX-64-STATIC: movl    $dcallee, %eax
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: daddress:
; LINUX-32-STATIC: 	movl	$dcallee, %eax
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: daddress:
; LINUX-32-PIC: 	movl	$dcallee, %eax
; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: daddress:
; LINUX-64-PIC: 	leaq	dcallee(%rip), %rax
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _daddress:
; DARWIN-32-STATIC: 	movl	$_dcallee, %eax
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _daddress:
; DARWIN-32-DYNAMIC: 	movl	$_dcallee, %eax
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _daddress:
; DARWIN-32-PIC: 	calll	L135$pb
; DARWIN-32-PIC-NEXT: L135$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	leal	_dcallee-L135$pb([[EAX]]), %eax
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _daddress:
; DARWIN-64-STATIC: 	leaq	_dcallee(%rip), %rax
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _daddress:
; DARWIN-64-DYNAMIC: 	leaq	_dcallee(%rip), %rax
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _daddress:
; DARWIN-64-PIC: 	leaq	_dcallee(%rip), %rax
; DARWIN-64-PIC-NEXT: 	ret
}

define void @caller() nounwind {
entry:
	call void @callee() nounwind
	call void @callee() nounwind
	ret void
; LINUX-64-STATIC-LABEL: caller:
; LINUX-64-STATIC: callq   callee
; LINUX-64-STATIC: callq   callee
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: caller:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	callee
; LINUX-32-STATIC-NEXT: 	calll	callee
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: caller:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	callee
; LINUX-32-PIC-NEXT: 	calll	callee
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: caller:
; LINUX-64-PIC: 	pushq
; LINUX-64-PIC-NEXT: 	callq	callee@PLT
; LINUX-64-PIC-NEXT: 	callq	callee@PLT
; LINUX-64-PIC-NEXT: 	popq
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _caller:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	_callee
; DARWIN-32-STATIC-NEXT: 	calll	_callee
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _caller:
; DARWIN-32-DYNAMIC: 	subl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	calll	_callee
; DARWIN-32-DYNAMIC-NEXT: 	calll	_callee
; DARWIN-32-DYNAMIC-NEXT: 	addl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _caller:
; DARWIN-32-PIC: 	subl	$12, %esp
; DARWIN-32-PIC-NEXT: 	calll	_callee
; DARWIN-32-PIC-NEXT: 	calll	_callee
; DARWIN-32-PIC-NEXT: 	addl	$12, %esp
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _caller:
; DARWIN-64-STATIC: 	pushq
; DARWIN-64-STATIC-NEXT: 	callq	_callee
; DARWIN-64-STATIC-NEXT: 	callq	_callee
; DARWIN-64-STATIC-NEXT: 	popq
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _caller:
; DARWIN-64-DYNAMIC: 	pushq
; DARWIN-64-DYNAMIC-NEXT: 	callq	_callee
; DARWIN-64-DYNAMIC-NEXT: 	callq	_callee
; DARWIN-64-DYNAMIC-NEXT: 	popq
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _caller:
; DARWIN-64-PIC: 	pushq
; DARWIN-64-PIC-NEXT: 	callq	_callee
; DARWIN-64-PIC-NEXT: 	callq	_callee
; DARWIN-64-PIC-NEXT: 	popq
; DARWIN-64-PIC-NEXT: 	ret
}

define void @dcaller() nounwind {
entry:
	call void @dcallee() nounwind
	call void @dcallee() nounwind
	ret void
; LINUX-64-STATIC-LABEL: dcaller:
; LINUX-64-STATIC: callq   dcallee
; LINUX-64-STATIC: callq   dcallee
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: dcaller:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	dcallee
; LINUX-32-STATIC-NEXT: 	calll	dcallee
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: dcaller:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	dcallee
; LINUX-32-PIC-NEXT: 	calll	dcallee
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: dcaller:
; LINUX-64-PIC: 	pushq
; LINUX-64-PIC-NEXT: 	callq	dcallee
; LINUX-64-PIC-NEXT: 	callq	dcallee
; LINUX-64-PIC-NEXT: 	popq
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _dcaller:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	_dcallee
; DARWIN-32-STATIC-NEXT: 	calll	_dcallee
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _dcaller:
; DARWIN-32-DYNAMIC: 	subl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	calll	_dcallee
; DARWIN-32-DYNAMIC-NEXT: 	calll	_dcallee
; DARWIN-32-DYNAMIC-NEXT: 	addl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _dcaller:
; DARWIN-32-PIC: 	subl	$12, %esp
; DARWIN-32-PIC-NEXT: 	calll	_dcallee
; DARWIN-32-PIC-NEXT: 	calll	_dcallee
; DARWIN-32-PIC-NEXT: 	addl	$12, %esp
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _dcaller:
; DARWIN-64-STATIC: 	pushq
; DARWIN-64-STATIC-NEXT: 	callq	_dcallee
; DARWIN-64-STATIC-NEXT: 	callq	_dcallee
; DARWIN-64-STATIC-NEXT: 	popq
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _dcaller:
; DARWIN-64-DYNAMIC: 	pushq
; DARWIN-64-DYNAMIC-NEXT: 	callq	_dcallee
; DARWIN-64-DYNAMIC-NEXT: 	callq	_dcallee
; DARWIN-64-DYNAMIC-NEXT: 	popq
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _dcaller:
; DARWIN-64-PIC: 	pushq
; DARWIN-64-PIC-NEXT: 	callq	_dcallee
; DARWIN-64-PIC-NEXT: 	callq	_dcallee
; DARWIN-64-PIC-NEXT: 	popq
; DARWIN-64-PIC-NEXT: 	ret
}

define void @lcaller() nounwind {
entry:
	call void @lcallee() nounwind
	call void @lcallee() nounwind
	ret void
; LINUX-64-STATIC-LABEL: lcaller:
; LINUX-64-STATIC: callq   lcallee
; LINUX-64-STATIC: callq   lcallee
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: lcaller:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	lcallee
; LINUX-32-STATIC-NEXT: 	calll	lcallee
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: lcaller:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	lcallee
; LINUX-32-PIC-NEXT: 	calll	lcallee
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: lcaller:
; LINUX-64-PIC: 	pushq
; LINUX-64-PIC-NEXT: 	callq	lcallee@PLT
; LINUX-64-PIC-NEXT: 	callq	lcallee@PLT
; LINUX-64-PIC-NEXT: 	popq
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _lcaller:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	_lcallee
; DARWIN-32-STATIC-NEXT: 	calll	_lcallee
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _lcaller:
; DARWIN-32-DYNAMIC: 	subl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	calll	_lcallee
; DARWIN-32-DYNAMIC-NEXT: 	calll	_lcallee
; DARWIN-32-DYNAMIC-NEXT: 	addl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _lcaller:
; DARWIN-32-PIC: 	subl	$12, %esp
; DARWIN-32-PIC-NEXT: 	calll	_lcallee
; DARWIN-32-PIC-NEXT: 	calll	_lcallee
; DARWIN-32-PIC-NEXT: 	addl	$12, %esp
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _lcaller:
; DARWIN-64-STATIC: 	pushq
; DARWIN-64-STATIC-NEXT: 	callq	_lcallee
; DARWIN-64-STATIC-NEXT: 	callq	_lcallee
; DARWIN-64-STATIC-NEXT: 	popq
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _lcaller:
; DARWIN-64-DYNAMIC: 	pushq
; DARWIN-64-DYNAMIC-NEXT: 	callq	_lcallee
; DARWIN-64-DYNAMIC-NEXT: 	callq	_lcallee
; DARWIN-64-DYNAMIC-NEXT: 	popq
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _lcaller:
; DARWIN-64-PIC: 	pushq
; DARWIN-64-PIC-NEXT: 	callq	_lcallee
; DARWIN-64-PIC-NEXT: 	callq	_lcallee
; DARWIN-64-PIC-NEXT: 	popq
; DARWIN-64-PIC-NEXT: 	ret
}

define void @tailcaller() nounwind {
entry:
	call void @callee() nounwind
	ret void
; LINUX-64-STATIC-LABEL: tailcaller:
; LINUX-64-STATIC: callq   callee
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: tailcaller:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	callee
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: tailcaller:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	callee
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: tailcaller:
; LINUX-64-PIC: 	pushq
; LINUX-64-PIC-NEXT: 	callq	callee@PLT
; LINUX-64-PIC-NEXT: 	popq
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _tailcaller:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	_callee
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _tailcaller:
; DARWIN-32-DYNAMIC: 	subl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	calll	_callee
; DARWIN-32-DYNAMIC-NEXT: 	addl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _tailcaller:
; DARWIN-32-PIC: 	subl	$12, %esp
; DARWIN-32-PIC-NEXT: 	calll	_callee
; DARWIN-32-PIC-NEXT: 	addl	$12, %esp
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _tailcaller:
; DARWIN-64-STATIC: 	pushq
; DARWIN-64-STATIC-NEXT: 	callq	_callee
; DARWIN-64-STATIC-NEXT: 	popq
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _tailcaller:
; DARWIN-64-DYNAMIC: 	pushq
; DARWIN-64-DYNAMIC-NEXT: 	callq	_callee
; DARWIN-64-DYNAMIC-NEXT: 	popq
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _tailcaller:
; DARWIN-64-PIC: 	pushq
; DARWIN-64-PIC-NEXT: 	callq	_callee
; DARWIN-64-PIC-NEXT: 	popq
; DARWIN-64-PIC-NEXT: 	ret
}

define void @dtailcaller() nounwind {
entry:
	call void @dcallee() nounwind
	ret void
; LINUX-64-STATIC-LABEL: dtailcaller:
; LINUX-64-STATIC: callq   dcallee
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: dtailcaller:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	dcallee
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: dtailcaller:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	dcallee
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: dtailcaller:
; LINUX-64-PIC: 	pushq
; LINUX-64-PIC-NEXT: 	callq	dcallee
; LINUX-64-PIC-NEXT: 	popq
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _dtailcaller:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	_dcallee
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _dtailcaller:
; DARWIN-32-DYNAMIC: 	subl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	calll	_dcallee
; DARWIN-32-DYNAMIC-NEXT: 	addl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _dtailcaller:
; DARWIN-32-PIC: 	subl	$12, %esp
; DARWIN-32-PIC-NEXT: 	calll	_dcallee
; DARWIN-32-PIC-NEXT: 	addl	$12, %esp
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _dtailcaller:
; DARWIN-64-STATIC: 	pushq
; DARWIN-64-STATIC-NEXT: 	callq	_dcallee
; DARWIN-64-STATIC-NEXT: 	popq
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _dtailcaller:
; DARWIN-64-DYNAMIC: 	pushq
; DARWIN-64-DYNAMIC-NEXT: 	callq	_dcallee
; DARWIN-64-DYNAMIC-NEXT: 	popq
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _dtailcaller:
; DARWIN-64-PIC: 	pushq
; DARWIN-64-PIC-NEXT: 	callq	_dcallee
; DARWIN-64-PIC-NEXT: 	popq
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ltailcaller() nounwind {
entry:
	call void @lcallee() nounwind
	ret void
; LINUX-64-STATIC-LABEL: ltailcaller:
; LINUX-64-STATIC: callq   lcallee
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ltailcaller:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	lcallee
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ltailcaller:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	lcallee
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ltailcaller:
; LINUX-64-PIC: 	pushq
; LINUX-64-PIC-NEXT: 	callq	lcallee@PLT
; LINUX-64-PIC-NEXT: 	popq
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ltailcaller:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	_lcallee
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ltailcaller:
; DARWIN-32-DYNAMIC: 	subl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	calll	_lcallee
; DARWIN-32-DYNAMIC-NEXT: 	addl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ltailcaller:
; DARWIN-32-PIC: 	subl	$12, %esp
; DARWIN-32-PIC-NEXT: 	calll	_lcallee
; DARWIN-32-PIC-NEXT: 	addl	$12, %esp
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ltailcaller:
; DARWIN-64-STATIC: 	pushq
; DARWIN-64-STATIC-NEXT: 	callq	_lcallee
; DARWIN-64-STATIC-NEXT: 	popq
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ltailcaller:
; DARWIN-64-DYNAMIC: 	pushq
; DARWIN-64-DYNAMIC-NEXT: 	callq	_lcallee
; DARWIN-64-DYNAMIC-NEXT: 	popq
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ltailcaller:
; DARWIN-64-PIC: 	pushq
; DARWIN-64-PIC-NEXT: 	callq	_lcallee
; DARWIN-64-PIC-NEXT: 	popq
; DARWIN-64-PIC-NEXT: 	ret
}

define void @icaller() nounwind {
entry:
	%0 = load void ()*, void ()** @ifunc, align 8
	call void %0() nounwind
	%1 = load void ()*, void ()** @ifunc, align 8
	call void %1() nounwind
	ret void
; LINUX-64-STATIC-LABEL: icaller:
; LINUX-64-STATIC: callq   *ifunc
; LINUX-64-STATIC: callq   *ifunc
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: icaller:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	*ifunc
; LINUX-32-STATIC-NEXT: 	calll	*ifunc
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: icaller:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	*ifunc
; LINUX-32-PIC-NEXT: 	calll	*ifunc
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: icaller:
; LINUX-64-PIC: 	pushq	[[RBX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	ifunc@GOTPCREL(%rip), [[RBX:%r.x]]
; LINUX-64-PIC-NEXT: 	callq	*([[RBX]])
; LINUX-64-PIC-NEXT: 	callq	*([[RBX]])
; LINUX-64-PIC-NEXT: 	popq	[[RBX:%r.x]]
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _icaller:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	*_ifunc
; DARWIN-32-STATIC-NEXT: 	calll	*_ifunc
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _icaller:
; DARWIN-32-DYNAMIC: 	pushl	%esi
; DARWIN-32-DYNAMIC-NEXT: 	subl	$8, %esp
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ifunc$non_lazy_ptr, %esi
; DARWIN-32-DYNAMIC-NEXT: 	calll	*(%esi)
; DARWIN-32-DYNAMIC-NEXT: 	calll	*(%esi)
; DARWIN-32-DYNAMIC-NEXT: 	addl	$8, %esp
; DARWIN-32-DYNAMIC-NEXT: 	popl	%esi
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _icaller:
; DARWIN-32-PIC: 	pushl	%esi
; DARWIN-32-PIC-NEXT: 	subl	$8, %esp
; DARWIN-32-PIC-NEXT: 	calll	L142$pb
; DARWIN-32-PIC-NEXT: L142$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ifunc$non_lazy_ptr-L142$pb([[EAX]]), %esi
; DARWIN-32-PIC-NEXT: 	calll	*(%esi)
; DARWIN-32-PIC-NEXT: 	calll	*(%esi)
; DARWIN-32-PIC-NEXT: 	addl	$8, %esp
; DARWIN-32-PIC-NEXT: 	popl	%esi
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _icaller:
; DARWIN-64-STATIC: 	pushq	[[RBX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ifunc@GOTPCREL(%rip), [[RBX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	callq	*([[RBX]])
; DARWIN-64-STATIC-NEXT: 	callq	*([[RBX]])
; DARWIN-64-STATIC-NEXT: 	popq	[[RBX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _icaller:
; DARWIN-64-DYNAMIC: 	pushq	[[RBX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ifunc@GOTPCREL(%rip), [[RBX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	callq	*([[RBX]])
; DARWIN-64-DYNAMIC-NEXT: 	callq	*([[RBX]])
; DARWIN-64-DYNAMIC-NEXT: 	popq	[[RBX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _icaller:
; DARWIN-64-PIC: 	pushq	[[RBX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ifunc@GOTPCREL(%rip), [[RBX:%r.x]]
; DARWIN-64-PIC-NEXT: 	callq	*([[RBX]])
; DARWIN-64-PIC-NEXT: 	callq	*([[RBX]])
; DARWIN-64-PIC-NEXT: 	popq	[[RBX:%r.x]]
; DARWIN-64-PIC-NEXT: 	ret
}

define void @dicaller() nounwind {
entry:
	%0 = load void ()*, void ()** @difunc, align 8
	call void %0() nounwind
	%1 = load void ()*, void ()** @difunc, align 8
	call void %1() nounwind
	ret void
; LINUX-64-STATIC-LABEL: dicaller:
; LINUX-64-STATIC: callq   *difunc
; LINUX-64-STATIC: callq   *difunc
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: dicaller:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	*difunc
; LINUX-32-STATIC-NEXT: 	calll	*difunc
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: dicaller:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	*difunc
; LINUX-32-PIC-NEXT: 	calll	*difunc
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: dicaller:
; LINUX-64-PIC: 	pushq	[[RBX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	difunc@GOTPCREL(%rip), [[RBX:%r.x]]
; LINUX-64-PIC-NEXT: 	callq	*([[RBX]])
; LINUX-64-PIC-NEXT: 	callq	*([[RBX]])
; LINUX-64-PIC-NEXT: 	popq	[[RBX:%r.x]]
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _dicaller:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	*_difunc
; DARWIN-32-STATIC-NEXT: 	calll	*_difunc
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _dicaller:
; DARWIN-32-DYNAMIC: 	subl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	calll	*_difunc
; DARWIN-32-DYNAMIC-NEXT: 	calll	*_difunc
; DARWIN-32-DYNAMIC-NEXT: 	addl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _dicaller:
; DARWIN-32-PIC: 	pushl	%esi
; DARWIN-32-PIC-NEXT: 	subl	$8, %esp
; DARWIN-32-PIC-NEXT: 	calll	L143$pb
; DARWIN-32-PIC-NEXT: L143$pb:
; DARWIN-32-PIC-NEXT: 	popl	%esi
; DARWIN-32-PIC-NEXT: 	calll	*_difunc-L143$pb(%esi)
; DARWIN-32-PIC-NEXT: 	calll	*_difunc-L143$pb(%esi)
; DARWIN-32-PIC-NEXT: 	addl	$8, %esp
; DARWIN-32-PIC-NEXT: 	popl	%esi
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _dicaller:
; DARWIN-64-STATIC: 	pushq
; DARWIN-64-STATIC-NEXT: 	callq	*_difunc(%rip)
; DARWIN-64-STATIC-NEXT: 	callq	*_difunc(%rip)
; DARWIN-64-STATIC-NEXT: 	popq
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _dicaller:
; DARWIN-64-DYNAMIC: 	pushq
; DARWIN-64-DYNAMIC-NEXT: 	callq	*_difunc(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	callq	*_difunc(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	popq
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _dicaller:
; DARWIN-64-PIC: 	pushq
; DARWIN-64-PIC-NEXT: 	callq	*_difunc(%rip)
; DARWIN-64-PIC-NEXT: 	callq	*_difunc(%rip)
; DARWIN-64-PIC-NEXT: 	popq
; DARWIN-64-PIC-NEXT: 	ret
}

define void @licaller() nounwind {
entry:
	%0 = load void ()*, void ()** @lifunc, align 8
	call void %0() nounwind
	%1 = load void ()*, void ()** @lifunc, align 8
	call void %1() nounwind
	ret void
; LINUX-64-STATIC-LABEL: licaller:
; LINUX-64-STATIC: callq   *lifunc
; LINUX-64-STATIC: callq   *lifunc
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: licaller:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	*lifunc
; LINUX-32-STATIC-NEXT: 	calll	*lifunc
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: licaller:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	*lifunc
; LINUX-32-PIC-NEXT: 	calll	*lifunc
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: licaller:
; LINUX-64-PIC: 	pushq
; LINUX-64-PIC-NEXT: 	callq	*lifunc(%rip)
; LINUX-64-PIC-NEXT: 	callq	*lifunc(%rip)
; LINUX-64-PIC-NEXT: 	popq
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _licaller:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	*_lifunc
; DARWIN-32-STATIC-NEXT: 	calll	*_lifunc
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _licaller:
; DARWIN-32-DYNAMIC: 	subl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	calll	*_lifunc
; DARWIN-32-DYNAMIC-NEXT: 	calll	*_lifunc
; DARWIN-32-DYNAMIC-NEXT: 	addl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _licaller:
; DARWIN-32-PIC: 	pushl	%esi
; DARWIN-32-PIC-NEXT: 	subl	$8, %esp
; DARWIN-32-PIC-NEXT: 	calll	L144$pb
; DARWIN-32-PIC-NEXT: L144$pb:
; DARWIN-32-PIC-NEXT: 	popl	%esi
; DARWIN-32-PIC-NEXT: 	calll	*_lifunc-L144$pb(%esi)
; DARWIN-32-PIC-NEXT: 	calll	*_lifunc-L144$pb(%esi)
; DARWIN-32-PIC-NEXT: 	addl	$8, %esp
; DARWIN-32-PIC-NEXT: 	popl	%esi
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _licaller:
; DARWIN-64-STATIC: 	pushq
; DARWIN-64-STATIC-NEXT: 	callq	*_lifunc(%rip)
; DARWIN-64-STATIC-NEXT: 	callq	*_lifunc(%rip)
; DARWIN-64-STATIC-NEXT: 	popq
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _licaller:
; DARWIN-64-DYNAMIC: 	pushq
; DARWIN-64-DYNAMIC-NEXT: 	callq	*_lifunc(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	callq	*_lifunc(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	popq
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _licaller:
; DARWIN-64-PIC: 	pushq
; DARWIN-64-PIC-NEXT: 	callq	*_lifunc(%rip)
; DARWIN-64-PIC-NEXT: 	callq	*_lifunc(%rip)
; DARWIN-64-PIC-NEXT: 	popq
; DARWIN-64-PIC-NEXT: 	ret
}

define void @itailcaller() nounwind {
entry:
	%0 = load void ()*, void ()** @ifunc, align 8
	call void %0() nounwind
	%1 = load void ()*, void ()** @ifunc, align 8
	call void %1() nounwind
	ret void
; LINUX-64-STATIC-LABEL: itailcaller:
; LINUX-64-STATIC: callq   *ifunc
; LINUX-64-STATIC: callq   *ifunc
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: itailcaller:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	*ifunc
; LINUX-32-STATIC-NEXT: 	calll	*ifunc
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: itailcaller:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	*ifunc
; LINUX-32-PIC-NEXT: 	calll	*ifunc
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: itailcaller:
; LINUX-64-PIC: 	pushq	[[RBX:%r.x]]
; LINUX-64-PIC-NEXT: 	movq	ifunc@GOTPCREL(%rip), [[RBX:%r.x]]
; LINUX-64-PIC-NEXT: 	callq	*([[RBX]])
; LINUX-64-PIC-NEXT: 	callq	*([[RBX]])
; LINUX-64-PIC-NEXT: 	popq	[[RBX:%r.x]]
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _itailcaller:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	*_ifunc
; DARWIN-32-STATIC-NEXT: 	calll	*_ifunc
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _itailcaller:
; DARWIN-32-DYNAMIC: 	pushl	%esi
; DARWIN-32-DYNAMIC-NEXT: 	subl	$8, %esp
; DARWIN-32-DYNAMIC-NEXT: 	movl	L_ifunc$non_lazy_ptr, %esi
; DARWIN-32-DYNAMIC-NEXT: 	calll	*(%esi)
; DARWIN-32-DYNAMIC-NEXT: 	calll	*(%esi)
; DARWIN-32-DYNAMIC-NEXT: 	addl	$8, %esp
; DARWIN-32-DYNAMIC-NEXT: 	popl	%esi
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _itailcaller:
; DARWIN-32-PIC: 	pushl	%esi
; DARWIN-32-PIC-NEXT: 	subl	$8, %esp
; DARWIN-32-PIC-NEXT: 	calll	L145$pb
; DARWIN-32-PIC-NEXT: L145$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	movl	L_ifunc$non_lazy_ptr-L145$pb([[EAX]]), %esi
; DARWIN-32-PIC-NEXT: 	calll	*(%esi)
; DARWIN-32-PIC-NEXT: 	calll	*(%esi)
; DARWIN-32-PIC-NEXT: 	addl	$8, %esp
; DARWIN-32-PIC-NEXT: 	popl	%esi
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _itailcaller:
; DARWIN-64-STATIC: 	pushq	[[RBX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	movq	_ifunc@GOTPCREL(%rip), [[RBX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	callq	*([[RBX]])
; DARWIN-64-STATIC-NEXT: 	callq	*([[RBX]])
; DARWIN-64-STATIC-NEXT: 	popq	[[RBX:%r.x]]
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _itailcaller:
; DARWIN-64-DYNAMIC: 	pushq	[[RBX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	movq	_ifunc@GOTPCREL(%rip), [[RBX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	callq	*([[RBX]])
; DARWIN-64-DYNAMIC-NEXT: 	callq	*([[RBX]])
; DARWIN-64-DYNAMIC-NEXT: 	popq	[[RBX:%r.x]]
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _itailcaller:
; DARWIN-64-PIC: 	pushq	[[RBX:%r.x]]
; DARWIN-64-PIC-NEXT: 	movq	_ifunc@GOTPCREL(%rip), [[RBX:%r.x]]
; DARWIN-64-PIC-NEXT: 	callq	*([[RBX]])
; DARWIN-64-PIC-NEXT: 	callq	*([[RBX]])
; DARWIN-64-PIC-NEXT: 	popq	[[RBX:%r.x]]
; DARWIN-64-PIC-NEXT: 	ret
}

define void @ditailcaller() nounwind {
entry:
	%0 = load void ()*, void ()** @difunc, align 8
	call void %0() nounwind
	ret void
; LINUX-64-STATIC-LABEL: ditailcaller:
; LINUX-64-STATIC: callq   *difunc
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: ditailcaller:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	*difunc
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: ditailcaller:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	*difunc
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: ditailcaller:
; LINUX-64-PIC: 	pushq
; LINUX-64-PIC-NEXT: 	movq	difunc@GOTPCREL(%rip), [[RAX:%r.x]]
; LINUX-64-PIC-NEXT: 	callq	*([[RAX]])
; LINUX-64-PIC-NEXT: 	popq
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _ditailcaller:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	*_difunc
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _ditailcaller:
; DARWIN-32-DYNAMIC: 	subl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	calll	*_difunc
; DARWIN-32-DYNAMIC-NEXT: 	addl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _ditailcaller:
; DARWIN-32-PIC: 	subl	$12, %esp
; DARWIN-32-PIC-NEXT: 	calll	L146$pb
; DARWIN-32-PIC-NEXT: L146$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	calll	*_difunc-L146$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	addl	$12, %esp
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _ditailcaller:
; DARWIN-64-STATIC: 	pushq
; DARWIN-64-STATIC-NEXT: 	callq	*_difunc(%rip)
; DARWIN-64-STATIC-NEXT: 	popq
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _ditailcaller:
; DARWIN-64-DYNAMIC: 	pushq
; DARWIN-64-DYNAMIC-NEXT: 	callq	*_difunc(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	popq
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _ditailcaller:
; DARWIN-64-PIC: 	callq	*_difunc(%rip)
; DARWIN-64-PIC-NEXT: 	popq
; DARWIN-64-PIC-NEXT: 	ret
}

define void @litailcaller() nounwind {
entry:
	%0 = load void ()*, void ()** @lifunc, align 8
	call void %0() nounwind
	ret void
; LINUX-64-STATIC-LABEL: litailcaller:
; LINUX-64-STATIC: callq   *lifunc
; LINUX-64-STATIC: ret

; LINUX-32-STATIC-LABEL: litailcaller:
; LINUX-32-STATIC: 	subl
; LINUX-32-STATIC-NEXT: 	calll	*lifunc
; LINUX-32-STATIC-NEXT: 	addl
; LINUX-32-STATIC-NEXT: 	ret

; LINUX-32-PIC-LABEL: litailcaller:
; LINUX-32-PIC: 	subl
; LINUX-32-PIC-NEXT: 	calll	*lifunc
; LINUX-32-PIC-NEXT: 	addl

; LINUX-32-PIC-NEXT: 	ret

; LINUX-64-PIC-LABEL: litailcaller:
; LINUX-64-PIC: 	pushq
; LINUX-64-PIC-NEXT: 	callq	*lifunc(%rip)
; LINUX-64-PIC-NEXT: 	popq
; LINUX-64-PIC-NEXT: 	ret

; DARWIN-32-STATIC: _litailcaller:
; DARWIN-32-STATIC: 	subl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	calll	*_lifunc
; DARWIN-32-STATIC-NEXT: 	addl	$12, %esp
; DARWIN-32-STATIC-NEXT: 	ret

; DARWIN-32-DYNAMIC: _litailcaller:
; DARWIN-32-DYNAMIC: 	subl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	calll	*_lifunc
; DARWIN-32-DYNAMIC-NEXT: 	addl	$12, %esp
; DARWIN-32-DYNAMIC-NEXT: 	ret

; DARWIN-32-PIC: _litailcaller:
; DARWIN-32-PIC: 	subl	$12, %esp
; DARWIN-32-PIC-NEXT: 	calll	L147$pb
; DARWIN-32-PIC-NEXT: L147$pb:
; DARWIN-32-PIC-NEXT: 	popl	[[EAX:%e.x]]
; DARWIN-32-PIC-NEXT: 	calll	*_lifunc-L147$pb([[EAX]])
; DARWIN-32-PIC-NEXT: 	addl	$12, %esp
; DARWIN-32-PIC-NEXT: 	ret

; DARWIN-64-STATIC: _litailcaller:
; DARWIN-64-STATIC: 	pushq
; DARWIN-64-STATIC-NEXT: 	callq	*_lifunc(%rip)
; DARWIN-64-STATIC-NEXT: 	popq
; DARWIN-64-STATIC-NEXT: 	ret

; DARWIN-64-DYNAMIC: _litailcaller:
; DARWIN-64-DYNAMIC: 	pushq
; DARWIN-64-DYNAMIC-NEXT: 	callq	*_lifunc(%rip)
; DARWIN-64-DYNAMIC-NEXT: 	popq
; DARWIN-64-DYNAMIC-NEXT: 	ret

; DARWIN-64-PIC: _litailcaller:
; DARWIN-64-PIC: 	pushq
; DARWIN-64-PIC-NEXT: 	callq	*_lifunc(%rip)
; DARWIN-64-PIC-NEXT: 	popq
; DARWIN-64-PIC-NEXT: 	ret
}
