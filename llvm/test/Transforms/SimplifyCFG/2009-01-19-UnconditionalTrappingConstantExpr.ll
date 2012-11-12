; RUN: opt < %s -simplifycfg -S | FileCheck %s
; PR3354
; Do not merge bb1 into the entry block, it might trap.

@G = extern_weak global i32

; CHECK: @test(
; CHECK: br i1 %tmp25
; CHECK: bb1:
; CHECK: sdiv

define i32 @test(i32 %tmp21, i32 %tmp24) {
	%tmp25 = icmp sle i32 %tmp21, %tmp24		
	br i1 %tmp25, label %bb2, label %bb1	
					
bb1:		; preds = %bb	
	%tmp26 = icmp sgt i32 sdiv (i32 -32768, i32 ptrtoint (i32* @G to i32)), 0
	br i1 %tmp26, label %bb6, label %bb2		
bb2:
	ret i32 42

bb6:
	ret i32 927
}

; CHECK: @test2(
; CHECK: br i1 %tmp34
; CHECK: bb5:
; CHECK: sdiv

define i32 @test2(i32 %tmp21, i32 %tmp24, i1 %tmp34) {
	br i1 %tmp34, label %bb5, label %bb6

bb5:		; preds = %bb4
	br i1 icmp sgt (i32 sdiv (i32 32767, i32 ptrtoint (i32* @G to i32)), i32 0), label %bb6, label %bb7
bb6:
	ret i32 42
bb7:
	ret i32 927
}

