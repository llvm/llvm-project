; RUN: llvm-as < %s | llc -march=x86 | not grep btl

; This tests some cases where BT must not be generated.  See also bt.ll.
; Fixes 20040709-[12].c in gcc testsuite.

define void @test2(i32 %x, i32 %n) nounwind {
entry:
        %tmp1 = and i32 %x, 1
        %tmp2 = urem i32 %tmp1, 15
	%tmp3 = and i32 %tmp2, 1		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, %tmp2	; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @test3(i32 %x, i32 %n) nounwind {
entry:
        %tmp1 = and i32 %x, 1
        %tmp2 = urem i32 %tmp1, 15
	%tmp3 = and i32 %tmp2, 1		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp2, %tmp3	; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @test4(i32 %x, i32 %n) nounwind {
entry:
        %tmp1 = and i32 %x, 1
        %tmp2 = urem i32 %tmp1, 15
	%tmp3 = and i32 %tmp2, 1		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp2, %tmp3	; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @test5(i32 %x, i32 %n) nounwind {
entry:
        %tmp1 = and i32 %x, 1
        %tmp2 = urem i32 %tmp1, 15
	%tmp3 = and i32 %tmp2, 1		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp2, %tmp3	; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

declare void @foo()
