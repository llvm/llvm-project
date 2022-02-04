; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; Check that code corresponding to the following C function is
; simplified into a single ASR operation:
;
; int test_asr(int a, int b) {
;   return a < 0 ? -(-a - 1 >> b) - 1 : a >> b;
; }
;
define i32 @test_asr(i32 %a, i32 %b) {
entry:
	%c = icmp slt i32 %a, 0
	br i1 %c, label %bb2, label %bb3

bb2:
	%t1 = sub i32 0, %a
	%not = sub i32 %t1, 1
	%d = ashr i32 %not, %b
	%t2 = sub i32 0, %d
	%not2 = sub i32 %t2, 1
	br label %bb4
bb3:
	%e = ashr i32 %a, %b
	br label %bb4
bb4:
        %f = phi i32 [ %not2, %bb2 ], [ %e, %bb3 ]
	ret i32 %f
; CHECK-LABEL: @test_asr(
; CHECK: bb4:
; CHECK: %f = ashr i32 %a, %b
; CHECK: ret i32 %f
}
