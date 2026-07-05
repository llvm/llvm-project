; RUN: llc < %s -mtriple=thumbv7-apple-darwin9 -mcpu=cortex-a8
; rdar://7117307

	%struct.Hosp = type { i32, i32, i32, %struct.List, %struct.List, %struct.List, %struct.List }
	%struct.List = type { ptr, ptr, ptr }
	%struct.Patient = type { i32, i32, i32, ptr }
	%struct.Village = type { [4 x ptr], ptr, %struct.List, %struct.Hosp, i32, i32 }

define ptr @sim(ptr %village) nounwind {
entry:
	br i1 undef, label %bb14, label %bb3.preheader

bb3.preheader:		; preds = %entry
	br label %bb5

bb5:		; preds = %bb5, %bb3.preheader
	br i1 undef, label %bb11, label %bb5

bb11:		; preds = %bb5
	%0 = fmul float undef, 0x41E0000000000000		; <float> [#uses=1]
	%1 = fptosi float %0 to i32		; <i32> [#uses=1]
	store i32 %1, ptr undef, align 4
	br i1 undef, label %generate_patient.exit, label %generate_patient.exit.thread

generate_patient.exit.thread:		; preds = %bb11
	ret ptr null

generate_patient.exit:		; preds = %bb11
	br i1 undef, label %bb14, label %bb12

bb12:		; preds = %generate_patient.exit
	br i1 undef, label %bb.i, label %bb1.i

bb.i:		; preds = %bb12
	ret ptr null

bb1.i:		; preds = %bb12
	ret ptr null

bb14:		; preds = %generate_patient.exit, %entry
	ret ptr undef
}
