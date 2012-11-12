; The loop canonicalization pass should guarantee that there is one backedge
; for all loops.  This allows the -indvars pass to recognize the %IV
; induction variable in this testcase.

; RUN: opt < %s -indvars -S | FileCheck %s
; CHECK: Loop.backedge:
; CHECK-NOT: br
; CHECK: br label %Loop

define i32 @test(i1 %C) {
; <label>:0
	br label %Loop
Loop:		; preds = %BE2, %BE1, %0
	%IV = phi i32 [ 1, %0 ], [ %IV2, %BE1 ], [ %IV2, %BE2 ]		; <i32> [#uses=2]
	store i32 %IV, i32* null
	%IV2 = add i32 %IV, 2		; <i32> [#uses=2]
	br i1 %C, label %BE1, label %BE2
BE1:		; preds = %Loop
	br label %Loop
BE2:		; preds = %Loop
	br label %Loop
}

