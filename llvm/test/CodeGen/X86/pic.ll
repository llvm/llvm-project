; RUN: llc < %s -mcpu=generic -mtriple=i686-pc-linux-gnu -relocation-model=pic -asm-verbose=false -post-RA-scheduler=false -verify-machineinstrs | FileCheck %s -check-prefixes=CHECK,CHECK-I686
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-pc-linux-gnux32 -relocation-model=pic -asm-verbose=false -post-RA-scheduler=false -verify-machineinstrs | FileCheck %s -check-prefixes=CHECK,CHECK-X32
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-pc-linux-gnux32 -relocation-model=pic -asm-verbose=false -post-RA-scheduler=false -fast-isel -verify-machineinstrs | FileCheck %s -check-prefixes=CHECK,CHECK-X32

@ptr = external global i32* 
@dst = external global i32 
@src = external global i32 

define void @test0() nounwind {
entry:
    store i32* @dst, i32** @ptr
    %tmp.s = load i32, i32* @src
    store i32 %tmp.s, i32* @dst
    ret void
    
; CHECK-LABEL:	test0:
; CHECK-I686:	calll	.L0$pb
; CHECK-I686-NEXT:	.L0$pb:
; CHECK-I686-NEXT:	popl
; CHECK-I686:	addl	$_GLOBAL_OFFSET_TABLE_+(.L{{.*}}-.L0$pb),
; CHECK-I686:	movl	dst@GOT(%eax),
; CHECK-I686:	movl	ptr@GOT(%eax),
; CHECK-I686:	movl	src@GOT(%eax),
; CHECK-I686:	ret
; CHECK-X32-DAG:	movl	dst@GOTPCREL(%rip),
; CHECK-X32-DAG:	movl	ptr@GOTPCREL(%rip),
; CHECK-X32-DAG:	movl	src@GOTPCREL(%rip),
; CHECK-X32:	retq
}

@ptr2 = global i32* null
@dst2 = global i32 0
@src2 = global i32 0

define void @test1() nounwind {
entry:
    store i32* @dst2, i32** @ptr2
    %tmp.s = load i32, i32* @src2
    store i32 %tmp.s, i32* @dst2
    ret void
    
; CHECK-LABEL:	test1:
; CHECK-I686:	calll	.L1$pb
; CHECK-I686-NEXT:	.L1$pb:
; CHECK-I686-NEXT:	popl
; CHECK-I686:	addl	$_GLOBAL_OFFSET_TABLE_+(.L{{.*}}-.L1$pb), %eax
; CHECK-I686:	movl	dst2@GOT(%eax),
; CHECK-I686:	movl	ptr2@GOT(%eax),
; CHECK-I686:	movl	src2@GOT(%eax),
; CHECK-I686:	ret
; CHECK-X32-DAG:	movl	dst2@GOTPCREL(%rip),
; CHECK-X32-DAG:	movl	ptr2@GOTPCREL(%rip),
; CHECK-X32-DAG:	movl	src2@GOTPCREL(%rip),
; CHECK-X32:	retq

}

declare i8* @malloc(i32)

define void @test2() nounwind {
entry:
    %ptr = call i8* @malloc(i32 40)
    ret void
; CHECK-LABEL:	test2:
; CHECK-I686:	pushl	%ebx
; CHECK-I686-NEXT:	subl	$8, %esp
; CHECK-I686-NEXT:	calll	.L2$pb
; CHECK-I686-NEXT:	.L2$pb:
; CHECK-I686-NEXT:	popl	%ebx
; CHECK-I686:	addl	$_GLOBAL_OFFSET_TABLE_+(.L{{.*}}-.L2$pb), %ebx
; CHECK-I686:	movl	$40, (%esp)
; CHECK-I686:	calll	malloc@PLT
; CHECK-I686:	addl	$8, %esp
; CHECK-I686:	popl	%ebx
; CHECK-I686:	ret
; CHECK-X32:	pushq	%rax
; CHECK-X32:	movl	$40, %edi
; CHECK-X32:	callq	malloc@PLT
; CHECK-X32:	popq	%rax
; CHECK-X32:	retq

}

@pfoo = external global void(...)* 

define void @test3() nounwind {
entry:
    %tmp = call void(...)*(...) @afoo()
    store void(...)* %tmp, void(...)** @pfoo
    %tmp1 = load void(...)*, void(...)** @pfoo
    call void(...) %tmp1()
    ret void
; CHECK-LABEL:	test3:
; CHECK-I686:	calll	.L3$pb
; CHECK-I686-NEXT:	.L3$pb:
; CHECK-I686:	popl
; CHECK-I686:	addl	$_GLOBAL_OFFSET_TABLE_+(.L{{.*}}-.L3$pb), %[[REG3:e..]]
; CHECK-I686:	calll	afoo@PLT
; CHECK-I686:	movl	pfoo@GOT(%[[REG3]]),
; CHECK-I686:	calll	*
; CHECK-X32:	callq	afoo@PLT
; CHECK-X32:	movl	pfoo@GOTPCREL(%rip),
; CHECK-X32:	callq	*
}

declare void(...)* @afoo(...)

define void @test4() nounwind {
entry:
    call void(...) @foo()
    ret void
; CHECK-LABEL:	test4:
; CHECK-I686:	calll	.L4$pb
; CHECK-I686:	popl	%ebx
; CHECK-I686:	addl	$_GLOBAL_OFFSET_TABLE_+(.L{{.*}}-.L4$pb), %ebx
; CHECK-I686:	calll	foo@PLT
; CHECK-X32:	callq	foo@PLT

}

declare void @foo(...)


@ptr6 = internal global i32* null
@dst6 = internal global i32 0
@src6 = internal global i32 0

define void @test5() nounwind {
entry:
    store i32* @dst6, i32** @ptr6
    %tmp.s = load i32, i32* @src6
    store i32 %tmp.s, i32* @dst6
    ret void
    
; CHECK-LABEL:	test5:
; CHECK-I686:	calll	.L5$pb
; CHECK-I686-NEXT:	.L5$pb:
; CHECK-I686-NEXT:	popl	%eax
; CHECK-I686:	addl	$_GLOBAL_OFFSET_TABLE_+(.L{{.*}}-.L5$pb), %eax
; CHECK-I686:	leal	dst6@GOTOFF(%eax), %ecx
; CHECK-I686:	movl	%ecx, ptr6@GOTOFF(%eax)
; CHECK-I686:	movl	src6@GOTOFF(%eax), %ecx
; CHECK-I686:	movl	%ecx, dst6@GOTOFF(%eax)
; CHECK-I686:	ret
; CHECK-X32:	leal	dst6(%rip), %eax
; CHECK-X32:	movl	%eax, ptr6(%rip)
; CHECK-X32:	movl	src6(%rip), %eax
; CHECK-X32:	movl	%eax, dst6(%rip)
; CHECK-X32:	retq
}


;; Test constant pool references.
define double @test6(i32 %a.u) nounwind {
entry:
    %tmp = icmp eq i32 %a.u,0
    %retval = select i1 %tmp, double 4.561230e+02, double 1.234560e+02
    ret double %retval

; CHECK:	.LCPI6_0:

; CHECK-LABEL:	test6:
; CHECK-I686:	calll .L6$pb
; CHECK-I686:	.L6$pb:
; CHECK-I686:	addl	$_GLOBAL_OFFSET_TABLE_+(.L{{.*}}-.L6$pb), 
; CHECK-I686:	fldl	.LCPI6_0@GOTOFF(
; CHECK-X32:		.LCPI6_0(%rip),
}


;; Test jump table references.
define void @test7(i32 %n.u) nounwind {
entry:
    switch i32 %n.u, label %bb12 [i32 1, label %bb i32 2, label %bb6 i32 4, label %bb7 i32 5, label %bb8 i32 6, label %bb10 i32 7, label %bb1 i32 8, label %bb3 i32 9, label %bb4 i32 10, label %bb9 i32 11, label %bb2 i32 12, label %bb5 i32 13, label %bb11 ]
bb:
    tail call void(...) @foo1()
    ret void
bb1:
    tail call void(...) @foo2()
    ret void
bb2:
    tail call void(...) @foo6()
    ret void
bb3:
    tail call void(...) @foo3()
    ret void
bb4:
    tail call void(...) @foo4()
    ret void
bb5:
    tail call void(...) @foo5()
    ret void
bb6:
    tail call void(...) @foo1()
    ret void
bb7:
    tail call void(...) @foo2()
    ret void
bb8:
    tail call void(...) @foo6()
    ret void
bb9:
    tail call void(...) @foo3()
    ret void
bb10:
    tail call void(...) @foo4()
    ret void
bb11:
    tail call void(...) @foo5()
    ret void
bb12:
    tail call void(...) @foo6()
    ret void
    
; CHECK-LABEL:	test7:
; CHECK-I686:	calll	.L7$pb
; CHECK-I686:	.L7$pb:
; CHECK-I686:	addl	$_GLOBAL_OFFSET_TABLE_+(.L{{.*}}-.L7$pb),
; CHECK-I686:	.LJTI7_0@GOTOFF(
; CHECK-I686:	jmpl	*
; CHECK-X32:	leal	.LJTI7_0(%rip), %eax
; CHECK-X32:	addl	(%eax,%edi,4), %eax
; CHECK-X32:	jmpq	*%rax

; CHECK:	.p2align 2
; CHECK-NEXT:	.LJTI7_0:
; CHECK-I686:	.long	 .LBB7_2@GOTOFF
; CHECK-I686:	.long	 .LBB7_8@GOTOFF
; CHECK-I686:	.long	 .LBB7_4@GOTOFF
; CHECK-I686:	.long	 .LBB7_6@GOTOFF
; CHECK-I686:	.long	 .LBB7_5@GOTOFF
; CHECK-I686:	.long	 .LBB7_8@GOTOFF
; CHECK-I686:	.long	 .LBB7_7@GOTOFF
; CHECK-X32:	.long	.LBB7_3-.LJTI7_0
; CHECK-X32:	.long	.LBB7_3-.LJTI7_0
; CHECK-X32:	.long	.LBB7_12-.LJTI7_0
; CHECK-X32:	.long	.LBB7_8-.LJTI7_0
; CHECK-X32:	.long	.LBB7_12-.LJTI7_0
; CHECK-X32:	.long	.LBB7_10-.LJTI7_0
; CHECK-X32:	.long	.LBB7_8-.LJTI7_0
; CHECK-X32:	.long	.LBB7_9-.LJTI7_0
; CHECK-X32:	.long	.LBB7_10-.LJTI7_0
; CHECK-X32:	.long	.LBB7_9-.LJTI7_0
; CHECK-X32:	.long	.LBB7_12-.LJTI7_0
; CHECK-X32:	.long	.LBB7_14-.LJTI7_0
; CHECK-X32:	.long	.LBB7_14-.LJTI7_0
}

declare void @foo1(...)
declare void @foo2(...)
declare void @foo6(...)
declare void @foo3(...)
declare void @foo4(...)
declare void @foo5(...)

;; Check TLS references
@tlsptrgd = thread_local global i32* null
@tlsdstgd = thread_local global i32 0
@tlssrcgd = thread_local global i32 0
@tlsptrld = thread_local(localdynamic) global i32* null
@tlsdstld = thread_local(localdynamic) global i32 0
@tlssrcld = thread_local(localdynamic) global i32 0
@tlsptrie = thread_local(initialexec) global i32* null
@tlsdstie = thread_local(initialexec) global i32 0
@tlssrcie = thread_local(initialexec) global i32 0
@tlsptrle = thread_local(localexec) global i32* null
@tlsdstle = thread_local(localexec) global i32 0
@tlssrcle = thread_local(localexec) global i32 0

define void @test8() nounwind {
entry:
    store i32* @tlsdstgd, i32** @tlsptrgd
    %tmp.s = load i32, i32* @tlssrcgd
    store i32 %tmp.s, i32* @tlsdstgd
    ret void

; CHECK-LABEL:	test8:
; CHECK-I686:	calll	.L8$pb
; CHECK-I686-NEXT:	.L8$pb:
; CHECK-I686-NEXT:	popl
; CHECK-I686:	addl	$_GLOBAL_OFFSET_TABLE_+(.L{{.*}}-.L8$pb), %ebx
; CHECK-I686-DAG:	leal	tlsdstgd@TLSGD(,%ebx), %eax
; CHECK-I686-DAG:	calll	___tls_get_addr@PLT
; CHECK-I686-DAG:	leal	tlsptrgd@TLSGD(,%ebx), %eax
; CHECK-I686-DAG:	calll	___tls_get_addr@PLT
; CHECK-I686-DAG:	leal	tlssrcgd@TLSGD(,%ebx), %eax
; CHECK-I686-DAG:	calll	___tls_get_addr@PLT
; CHECK-X32-DAG:	leaq	tlsdstgd@TLSGD(%rip), %rdi
; CHECK-X32-DAG:	callq	__tls_get_addr@PLT
; CHECK-X32-DAG:	leaq	tlsptrgd@TLSGD(%rip), %rdi
; CHECK-X32-DAG:	callq	__tls_get_addr@PLT
; CHECK-X32-DAG:	leaq	tlssrcgd@TLSGD(%rip), %rdi
; CHECK-X32-DAG:	callq	__tls_get_addr@PLT
; CHECK-I686:	ret
; CHECK-X32:	retq
}

define void @test9() nounwind {
entry:
    store i32* @tlsdstld, i32** @tlsptrld
    %tmp.s = load i32, i32* @tlssrcld
    store i32 %tmp.s, i32* @tlsdstld
    ret void

; CHECK-LABEL:	test9:
; CHECK-I686:	calll	.L9$pb
; CHECK-I686-NEXT:	.L9$pb:
; CHECK-I686-NEXT:	popl
; CHECK-I686:	addl	$_GLOBAL_OFFSET_TABLE_+(.L{{.*}}-.L9$pb), %ebx
; CHECK-I686:	leal	tlsdstld@TLSLDM(%ebx), %eax
; CHECK-X32:	leaq	tlsdstld@TLSLD(%rip), %rdi
; CHECK-I686:	calll	___tls_get_addr@PLT
; CHECK-X32:	callq	__tls_get_addr@PLT
; CHECK:	leal	tlsdstld@DTPOFF(
; CHECK:	movl	{{%.*}}, tlsptrld@DTPOFF(
; CHECK:	movl	tlssrcld@DTPOFF(
; CHECK:	movl	{{%.*}}, tlsdstld@DTPOFF(
; CHECK-I686:	ret
; CHECK-X32:	retq
}

define void @test10() nounwind {
entry:
    store i32* @tlsdstie, i32** @tlsptrie
    %tmp.s = load i32, i32* @tlssrcie
    store i32 %tmp.s, i32* @tlsdstie
    ret void

; CHECK-LABEL:	test10:
; CHECK-I686:	calll	.L10$pb
; CHECK-I686-NEXT:	.L10$pb:
; CHECK-I686-NEXT:	popl
; CHECK-I686:	addl	$_GLOBAL_OFFSET_TABLE_+(.L{{.*}}-.L10$pb),
; CHECK-I686-DAG:	movl	tlsdstie@GOTNTPOFF(
; CHECK-I686-DAG:	movl	%gs:0,
; CHECK-X32-DAG:	movl	tlsdstie@GOTTPOFF(%rip),
; CHECK-X32-DAG:	movl	%fs:0,
; CHECK:	addl
; CHECK-I686:	movl	tlsptrie@GOTNTPOFF(
; CHECK-X32:	movl	tlsptrie@GOTTPOFF(%rip),
; CHECK-I686:	movl	{{%.*}}, %gs:(
; CHECK-X32:	movl	{{%.*}}, %fs:(
; CHECK-I686:	movl	tlssrcie@GOTNTPOFF(
; CHECK-X32:	movl	tlssrcie@GOTTPOFF(%rip),
; CHECK-I686:	movl	%gs:(
; CHECK-X32:	movl	%fs:(
; CHECK-I686:	movl	{{%.*}}, %gs:(
; CHECK-X32:	movl	{{%.*}}, %fs:(
; CHECK-I686:	ret
; CHECK-X32:	retq
}

define void @test11() nounwind {
entry:
    store i32* @tlsdstle, i32** @tlsptrle
    %tmp.s = load i32, i32* @tlssrcle
    store i32 %tmp.s, i32* @tlsdstle
    ret void

; CHECK-LABEL:	test11:
; CHECK-I686:	movl	%gs:0,
; CHECK-X32:	movl	%fs:0,
; CHECK-I686:	leal	tlsdstle@NTPOFF(
; CHECK-X32:	leal	tlsdstle@TPOFF(
; CHECK-I686:	movl	{{%.*}}, %gs:tlsptrle@NTPOFF
; CHECK-X32:	movl	{{%.*}}, %fs:tlsptrle@TPOFF
; CHECK-I686:	movl	%gs:tlssrcle@NTPOFF,
; CHECK-X32:	movl	%fs:tlssrcle@TPOFF,
; CHECK-I686:	movl	{{%.*}}, %gs:tlsdstle@NTPOFF
; CHECK-X32:	movl	{{%.*}}, %fs:tlsdstle@TPOFF
; CHECK-I686:	ret
; CHECK-X32:	retq
}
