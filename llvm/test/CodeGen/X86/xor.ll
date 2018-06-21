; RUN: llc < %s -mtriple=i686-unknown -mattr=+sse2 | FileCheck %s -check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-linux -mattr=+sse2 | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-win32 -mattr=+sse2 | FileCheck %s -check-prefix=X64

; Though it is undefined, we want xor undef,undef to produce zero.
define <4 x i32> @test1() nounwind {
	%tmp = xor <4 x i32> undef, undef
	ret <4 x i32> %tmp
        
; X32-LABEL: test1:
; X32:	xorps	%xmm0, %xmm0
; X32:	ret
}

; Though it is undefined, we want xor undef,undef to produce zero.
define i32 @test2() nounwind{
	%tmp = xor i32 undef, undef
	ret i32 %tmp
; X32-LABEL: test2:
; X32:	xorl	%eax, %eax
; X32:	ret
}

define i32 @test3(i32 %a, i32 %b) nounwind  {
entry:
        %tmp1not = xor i32 %b, -2
	%tmp3 = and i32 %tmp1not, %a
        %tmp4 = lshr i32 %tmp3, 1
        ret i32 %tmp4
        
; X64-LABEL: test3:
; X64:	notl
; X64:	andl
; X64:	shrl
; X64:	ret

; X32-LABEL: test3:
; X32: 	movl	8(%esp), %eax
; X32: 	notl	%eax
; X32: 	andl	4(%esp), %eax
; X32: 	shrl	%eax
; X32: 	ret
}

define i32 @test4(i32 %a, i32 %b) nounwind  {
entry:
        br label %bb
bb:
	%b_addr.0 = phi i32 [ %b, %entry ], [ %tmp8, %bb ]
        %a_addr.0 = phi i32 [ %a, %entry ], [ %tmp3, %bb ]
	%tmp3 = xor i32 %a_addr.0, %b_addr.0
        %tmp4not = xor i32 %tmp3, 2147483647
        %tmp6 = and i32 %tmp4not, %b_addr.0
        %tmp8 = shl i32 %tmp6, 1
        %tmp10 = icmp eq i32 %tmp8, 0
	br i1 %tmp10, label %bb12, label %bb
bb12:
	ret i32 %tmp3
        
; X64-LABEL: test4:
; X64:    notl	[[REG:%[a-z]+]]
; X64:    andl	{{.*}}[[REG]]
; X32-LABEL: test4:
; X32:    notl	[[REG:%[a-z]+]]
; X32:    andl	{{.*}}[[REG]]
}

define i16 @test5(i16 %a, i16 %b) nounwind  {
entry:
        br label %bb
bb:
	%b_addr.0 = phi i16 [ %b, %entry ], [ %tmp8, %bb ]
        %a_addr.0 = phi i16 [ %a, %entry ], [ %tmp3, %bb ]
	%tmp3 = xor i16 %a_addr.0, %b_addr.0
        %tmp4not = xor i16 %tmp3, 32767
        %tmp6 = and i16 %tmp4not, %b_addr.0
        %tmp8 = shl i16 %tmp6, 1
        %tmp10 = icmp eq i16 %tmp8, 0
	br i1 %tmp10, label %bb12, label %bb
bb12:
	ret i16 %tmp3
; X64-LABEL: test5:
; X64:    notl	[[REG:%[a-z]+]]
; X64:    andl	{{.*}}[[REG]]
; X32-LABEL: test5:
; X32:    notl	[[REG:%[a-z]+]]
; X32:    andl	{{.*}}[[REG]]
}

define i8 @test6(i8 %a, i8 %b) nounwind  {
entry:
        br label %bb
bb:
	%b_addr.0 = phi i8 [ %b, %entry ], [ %tmp8, %bb ]
        %a_addr.0 = phi i8 [ %a, %entry ], [ %tmp3, %bb ]
	%tmp3 = xor i8 %a_addr.0, %b_addr.0
        %tmp4not = xor i8 %tmp3, 127
        %tmp6 = and i8 %tmp4not, %b_addr.0
        %tmp8 = shl i8 %tmp6, 1
        %tmp10 = icmp eq i8 %tmp8, 0
	br i1 %tmp10, label %bb12, label %bb
bb12:
	ret i8 %tmp3
; X64-LABEL: test6:
; X64:    notb	[[REG:%[a-z]+]]
; X64:    andb	{{.*}}[[REG]]
; X32-LABEL: test6:
; X32:    notb	[[REG:%[a-z]+]]
; X32:    andb	{{.*}}[[REG]]
}

define i32 @test7(i32 %a, i32 %b) nounwind  {
entry:
        br label %bb
bb:
	%b_addr.0 = phi i32 [ %b, %entry ], [ %tmp8, %bb ]
        %a_addr.0 = phi i32 [ %a, %entry ], [ %tmp3, %bb ]
	%tmp3 = xor i32 %a_addr.0, %b_addr.0
        %tmp4not = xor i32 %tmp3, 2147483646
        %tmp6 = and i32 %tmp4not, %b_addr.0
        %tmp8 = shl i32 %tmp6, 1
        %tmp10 = icmp eq i32 %tmp8, 0
	br i1 %tmp10, label %bb12, label %bb
bb12:
	ret i32 %tmp3
; X64-LABEL: test7:
; X64:    xorl	$2147483646, [[REG:%[a-z]+]]
; X64:    andl	{{.*}}[[REG]]
; X32-LABEL: test7:
; X32:    xorl	$2147483646, [[REG:%[a-z]+]]
; X32:    andl	{{.*}}[[REG]]
}

define i32 @test8(i32 %a) nounwind {
; rdar://7553032
entry:
  %t1 = sub i32 0, %a
  %t2 = add i32 %t1, -1
  ret i32 %t2
; X64-LABEL: test8:
; X64:   notl {{%eax|%edi|%ecx}}
; X32-LABEL: test8:
; X32:   notl %eax
}

define i32 @test9(i32 %a) nounwind {
  %1 = and i32 %a, 4096
  %2 = xor i32 %1, 4096
  ret i32 %2
; X64-LABEL: test9:
; X64:    notl	[[REG:%[a-z]+]]
; X64:    andl	{{.*}}[[REG:%[a-z]+]]
; X32-LABEL: test9:
; X32:    notl	[[REG:%[a-z]+]]
; X32:    andl	{{.*}}[[REG:%[a-z]+]]
}

; PR15948
define <4 x i32> @test10(<4 x i32> %a) nounwind {
  %1 = and <4 x i32> %a, <i32 4096, i32 4096, i32 4096, i32 4096>
  %2 = xor <4 x i32> %1, <i32 4096, i32 4096, i32 4096, i32 4096>
  ret <4 x i32> %2
; X64-LABEL: test10:
; X64:    andnps
; X32-LABEL: test10:
; X32:    andnps
}

define i32 @PR17487(i1 %tobool) {
  %tmp = insertelement <2 x i1> undef, i1 %tobool, i32 1
  %tmp1 = zext <2 x i1> %tmp to <2 x i64>
  %tmp2 = xor <2 x i64> %tmp1, <i64 1, i64 1>
  %tmp3 = extractelement <2 x i64> %tmp2, i32 1
  %add = add nsw i64 0, %tmp3
  %cmp6 = icmp ne i64 %add, 1
  %conv7 = zext i1 %cmp6 to i32
  ret i32 %conv7

; X64-LABEL: PR17487:
; X64: andn
; X32-LABEL: PR17487:
; X32: andn
}

define i32 @test11(i32 %b) {
  %shl = shl i32 1, %b
  %neg = xor i32 %shl, -1
  ret i32 %neg
; X64-LABEL: test11:
; X64: movl    $-2, %[[REG:.*]]
; X64: roll    %{{.*}}, %[[REG]]
; X32-LABEL: test11:
; X32: movl    $-2, %[[REG:.*]]
; X32: roll    %{{.*}}, %[[REG]]
}

%struct.ref_s = type { %union.v, i16, i16 }
%union.v = type { i64 }

define %struct.ref_s* @test12(%struct.ref_s* %op, i64 %osbot, i64 %intval) {
  %neg = shl i64 %intval, 32
  %sext = xor i64 %neg, -4294967296
  %idx.ext = ashr exact i64 %sext, 32
  %add.ptr = getelementptr inbounds %struct.ref_s, %struct.ref_s* %op, i64 %idx.ext
  ret %struct.ref_s* %add.ptr
; X64-LABEL: test12:
; X64: shlq	$32, %[[REG:.*]]
; X64-NOT: not
; X64: sarq	$28, %[[REG]]
; X32-LABEL: test12:
; X32: leal
; X32-NOT: not
; X32: shll	$2, %eax
}
