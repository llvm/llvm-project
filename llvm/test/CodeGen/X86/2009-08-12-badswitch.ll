; RUN: llc < %s | grep LJT
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10"

declare void @f1() nounwind readnone
declare void @f2() nounwind readnone
declare void @f3() nounwind readnone
declare void @f4() nounwind readnone
declare void @f5() nounwind readnone
declare void @f6() nounwind readnone
declare void @f7() nounwind readnone
declare void @f8() nounwind readnone
declare void @f9() nounwind readnone
declare void @f10() nounwind readnone
declare void @f11() nounwind readnone
declare void @f12() nounwind readnone
declare void @f13() nounwind readnone
declare void @f14() nounwind readnone
declare void @f15() nounwind readnone
declare void @f16() nounwind readnone
declare void @f17() nounwind readnone
declare void @f18() nounwind readnone
declare void @f19() nounwind readnone
declare void @f20() nounwind readnone
declare void @f21() nounwind readnone
declare void @f22() nounwind readnone
declare void @f23() nounwind readnone
declare void @f24() nounwind readnone
declare void @f25() nounwind readnone
declare void @f26() nounwind readnone

define internal fastcc i32 @foo(i64 %bar) nounwind ssp {
entry:
        br label %bb49

bb49:
	switch i64 %bar, label %RETURN [
		i64 2, label %RRETURN_2
		i64 3, label %RRETURN_6
		i64 4, label %RRETURN_7
		i64 5, label %RRETURN_14
		i64 6, label %RRETURN_15
		i64 7, label %RRETURN_16
		i64 8, label %RRETURN_17
		i64 9, label %RRETURN_18
		i64 10, label %RRETURN_19
		i64 11, label %RRETURN_20
		i64 12, label %RRETURN_21
		i64 13, label %RRETURN_22
		i64 14, label %RRETURN_24
		i64 15, label %RRETURN_26
		i64 16, label %RRETURN_27
		i64 17, label %RRETURN_28
		i64 18, label %RRETURN_29
		i64 19, label %RRETURN_30
		i64 20, label %RRETURN_31
		i64 21, label %RRETURN_38
		i64 22, label %RRETURN_40
		i64 23, label %RRETURN_42
		i64 24, label %RRETURN_44
		i64 25, label %RRETURN_48
		i64 26, label %RRETURN_52
		i64 27, label %RRETURN_1
	]

RETURN:
        call void @f1()
        br label %EXIT

RRETURN_2:		; preds = %bb49
        call void @f2()
        br label %EXIT

RRETURN_6:		; preds = %bb49
        call void @f2()
        br label %EXIT

RRETURN_7:		; preds = %bb49
        call void @f3()
        br label %EXIT

RRETURN_14:		; preds = %bb49
        call void @f4()
        br label %EXIT

RRETURN_15:		; preds = %bb49
        call void @f5()
        br label %EXIT

RRETURN_16:		; preds = %bb49
        call void @f6()
        br label %EXIT

RRETURN_17:		; preds = %bb49
        call void @f7()
        br label %EXIT

RRETURN_18:		; preds = %bb49
        call void @f8()
        br label %EXIT

RRETURN_19:		; preds = %bb49
        call void @f9()
        br label %EXIT

RRETURN_20:		; preds = %bb49
        call void @f10()
        br label %EXIT

RRETURN_21:		; preds = %bb49
        call void @f11()
        br label %EXIT

RRETURN_22:		; preds = %bb49
        call void @f12()
        br label %EXIT

RRETURN_24:		; preds = %bb49
        call void @f13()
        br label %EXIT

RRETURN_26:		; preds = %bb49
        call void @f14()
        br label %EXIT

RRETURN_27:		; preds = %bb49
        call void @f15()
        br label %EXIT

RRETURN_28:		; preds = %bb49
        call void @f16()
        br label %EXIT

RRETURN_29:		; preds = %bb49
        call void @f17()
        br label %EXIT

RRETURN_30:		; preds = %bb49
        call void @f18()
        br label %EXIT

RRETURN_31:		; preds = %bb49
        call void @f19()
        br label %EXIT

RRETURN_38:		; preds = %bb49
        call void @f20()
        br label %EXIT

RRETURN_40:		; preds = %bb49
        call void @f21()
        br label %EXIT

RRETURN_42:		; preds = %bb49
        call void @f22()
        br label %EXIT

RRETURN_44:		; preds = %bb49
        call void @f23()
        br label %EXIT

RRETURN_48:		; preds = %bb49
        call void @f24()
        br label %EXIT

RRETURN_52:		; preds = %bb49
        call void @f25()
        br label %EXIT

RRETURN_1:		; preds = %bb49
        call void @f26()
        br label %EXIT

EXIT:
        ret i32 0
}
