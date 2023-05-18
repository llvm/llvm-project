; RUN: llc < %s -mtriple=i386-linux-gnu -mcpu=penryn -mattr=sse4.1 | FileCheck %s --check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-linux -mcpu=penryn -mattr=sse4.1 | FileCheck %s --check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-win32 -mcpu=penryn -mattr=sse4.1 | FileCheck %s --check-prefix=X64

define i32 @test1() nounwind readonly {
; X32-LABEL: test1:
; X32:       # %bb.0: # %entry
; X32-NEXT:    movl %gs:196, %eax
; X32-NEXT:    movl (%eax), %eax
; X32-NEXT:    retl
;
; X64-LABEL: test1:
; X64:       # %bb.0: # %entry
; X64-NEXT:    movq %gs:320, %rax
; X64-NEXT:    movl (%rax), %eax
; X64-NEXT:    retq
entry:
	%tmp = load ptr, ptr addrspace(256) getelementptr (ptr, ptr addrspace(256) inttoptr (i32 72 to ptr addrspace(256)), i32 31)		; <ptr> [#uses=1]
	%tmp1 = load i32, ptr %tmp		; <i32> [#uses=1]
	ret i32 %tmp1
}

define i64 @test2(ptr addrspace(256) %tmp8) nounwind {
; X32-LABEL: test2:
; X32:       # %bb.0: # %entry
; X32-NEXT:    subl $12, %esp
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    calll *%gs:(%eax)
; X32-NEXT:    xorl %eax, %eax
; X32-NEXT:    xorl %edx, %edx
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl
;
; X64-LABEL: test2:
; X64:       # %bb.0: # %entry
; X64-NEXT:    {{(subq.*%rsp|pushq)}}
; X64-NEXT:    callq *%gs:(%{{(rcx|rdi)}})
; X64-NEXT:    xorl %eax, %eax
; X64-NEXT:    {{(addq.*%rsp|popq)}}
; X64-NEXT:    retq
entry:
  %tmp9 = load ptr, ptr addrspace(256) %tmp8, align 8
  tail call void %tmp9(ptr undef) nounwind optsize
  ret i64 0
}

define <2 x i64> @pmovsxwd_1(ptr addrspace(256) %p) nounwind readonly {
; X32-LABEL: pmovsxwd_1:
; X32:       # %bb.0: # %entry
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    pmovsxwd %gs:(%eax), %xmm0
; X32-NEXT:    retl
;
; X64-LABEL: pmovsxwd_1:
; X64:       # %bb.0: # %entry
; X64-NEXT:    pmovsxwd %gs:(%{{(rcx|rdi)}}), %xmm0
; X64-NEXT:    retq
entry:
  %0 = load i64, ptr addrspace(256) %p
  %tmp2 = insertelement <2 x i64> zeroinitializer, i64 %0, i32 0
  %1 = bitcast <2 x i64> %tmp2 to <8 x i16>
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = sext <4 x i16> %2 to <4 x i32>
  %4 = bitcast <4 x i32> %3 to <2 x i64>
  ret <2 x i64> %4
}

; The two loads here both look identical to selection DAG, except for their
; address spaces.  Make sure they aren't CSE'd.
define i32 @test_no_cse() nounwind readonly {
; X32-LABEL: test_no_cse:
; X32:       # %bb.0: # %entry
; X32-NEXT:    movl %gs:196, %eax
; X32-NEXT:    movl (%eax), %eax
; X32-NEXT:    movl %fs:196, %ecx
; X32-NEXT:    addl (%ecx), %eax
; X32-NEXT:    retl
;
; X64-LABEL: test_no_cse:
; X64:       # %bb.0: # %entry
; X64-NEXT:    movq %gs:320, %rax
; X64-NEXT:    movl (%rax), %eax
; X64-NEXT:    movq %fs:320, %rcx
; X64-NEXT:    addl (%rcx), %eax
; X64-NEXT:    retq
entry:
	%tmp = load ptr, ptr addrspace(256) getelementptr (ptr, ptr addrspace(256) inttoptr (i32 72 to ptr addrspace(256)), i32 31)		; <ptr> [#uses=1]
	%tmp1 = load i32, ptr %tmp		; <i32> [#uses=1]
	%tmp2 = load ptr, ptr addrspace(257) getelementptr (ptr, ptr addrspace(257) inttoptr (i32 72 to ptr addrspace(257)), i32 31)		; <ptr> [#uses=1]
	%tmp3 = load i32, ptr %tmp2		; <i32> [#uses=1]
	%tmp4 = add i32 %tmp1, %tmp3
	ret i32 %tmp4
}
