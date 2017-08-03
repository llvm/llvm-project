; RUN: llc < %s -mcpu=generic -march=x86 | FileCheck %s
	%struct._obstack_chunk = type { i8*, %struct._obstack_chunk*, [4 x i8] }
	%struct.obstack = type { i32, %struct._obstack_chunk*, i8*, i8*, i8*, i32, i32, %struct._obstack_chunk* (...)*, void (...)*, i8*, i8 }
@stmt_obstack = external global %struct.obstack		; <%struct.obstack*> [#uses=1]

; This should just not crash.
define void @test1() nounwind {
entry:
	br i1 true, label %cond_true, label %cond_next

cond_true:		; preds = %entry
	%new_size.0.i = select i1 false, i32 0, i32 0		; <i32> [#uses=1]
	%tmp.i = load i32, i32* bitcast (i8* getelementptr (%struct.obstack, %struct.obstack* @stmt_obstack, i32 0, i32 10) to i32*)		; <i32> [#uses=1]
	%tmp.i.upgrd.1 = trunc i32 %tmp.i to i8		; <i8> [#uses=1]
	%tmp21.i = and i8 %tmp.i.upgrd.1, 1		; <i8> [#uses=1]
	%tmp22.i = icmp eq i8 %tmp21.i, 0		; <i1> [#uses=1]
	br i1 %tmp22.i, label %cond_false30.i, label %cond_true23.i

cond_true23.i:		; preds = %cond_true
	ret void

cond_false30.i:		; preds = %cond_true
	%tmp35.i = tail call %struct._obstack_chunk* null( i32 %new_size.0.i )		; <%struct._obstack_chunk*> [#uses=0]
	ret void

cond_next:		; preds = %entry
	ret void
}



define i32 @test2(i16* %P, i16* %Q) nounwind {
  %A = load i16, i16* %P, align 4                      ; <i16> [#uses=11]
  %C = zext i16 %A to i32                         ; <i32> [#uses=1]
  %D = and i32 %C, 255                            ; <i32> [#uses=1]
  br label %L
L:

  store i16 %A, i16* %Q
  ret i32 %D

; CHECK-LABEL: test2:
; CHECK: 	movl	4(%esp), %eax
; CHECK-NEXT:	movzwl	(%eax), %e{{..}}

}

; rdar://10554090
; xor in exit block will be CSE'ed and load will be folded to xor in entry.
define i1 @test3(i32* %P, i32* %Q) nounwind {
; CHECK-LABEL: test3:
; CHECK: movl 8(%esp), %e
; CHECK: movl 4(%esp), %e
; CHECK: xorl (%e
; CHECK: j
entry:
  %0 = load i32, i32* %P, align 4
  %1 = load i32, i32* %Q, align 4
  %2 = xor i32 %0, %1
  %3 = and i32 %2, 89947
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %exit, label %land.end

exit:
  %shr.i.i19 = xor i32 %1, %0
  %5 = and i32 %shr.i.i19, 3456789123
  %6 = icmp eq i32 %5, 0
  br label %land.end

land.end:
  %7 = phi i1 [ %6, %exit ], [ false, %entry ]
  ret i1 %7
}
