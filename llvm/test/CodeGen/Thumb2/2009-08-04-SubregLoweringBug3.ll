; RUN: llc < %s -mtriple=thumbv7-apple-darwin9 -mcpu=cortex-a8
; rdar://7117307

	%struct.Hosp = type { i32, i32, i32, %struct.List, %struct.List, %struct.List, %struct.List }
	%struct.List = type { %struct.List*, %struct.Patient*, %struct.List* }
	%struct.Patient = type { i32, i32, i32, %struct.Village* }
	%struct.Village = type { [4 x %struct.Village*], %struct.Village*, %struct.List, %struct.Hosp, i32, i32 }

define %struct.List* @sim(%struct.Village* %village) nounwind {
entry:
	br i1 undef, label %bb14, label %bb3.preheader

bb3.preheader:		; preds = %entry
	br label %bb5

bb5:		; preds = %bb5, %bb3.preheader
	br i1 undef, label %bb11, label %bb5

bb11:		; preds = %bb5
	%0 = load i32* undef, align 4		; <i32> [#uses=1]
	%1 = xor i32 %0, 123459876		; <i32> [#uses=1]
	%2 = sdiv i32 %1, 127773		; <i32> [#uses=1]
	%3 = mul i32 %2, 2836		; <i32> [#uses=1]
	%4 = sub i32 0, %3		; <i32> [#uses=1]
	%5 = xor i32 %4, 123459876		; <i32> [#uses=1]
	%idum_addr.0.i.i = select i1 undef, i32 undef, i32 %5		; <i32> [#uses=1]
	%6 = sitofp i32 %idum_addr.0.i.i to double		; <double> [#uses=1]
	%7 = fmul double %6, 0x3E00000000200000		; <double> [#uses=1]
	%8 = fptrunc double %7 to float		; <float> [#uses=2]
	%9 = fmul float %8, 0x41E0000000000000		; <float> [#uses=1]
	%10 = fptosi float %9 to i32		; <i32> [#uses=1]
	store i32 %10, i32* undef, align 4
	%11 = fpext float %8 to double		; <double> [#uses=1]
	%12 = fcmp ogt double %11, 6.660000e-01		; <i1> [#uses=1]
	br i1 %12, label %generate_patient.exit, label %generate_patient.exit.thread

generate_patient.exit.thread:		; preds = %bb11
	ret %struct.List* null

generate_patient.exit:		; preds = %bb11
	br i1 undef, label %bb14, label %bb12

bb12:		; preds = %generate_patient.exit
	br i1 undef, label %bb.i, label %bb1.i

bb.i:		; preds = %bb12
	ret %struct.List* null

bb1.i:		; preds = %bb12
	ret %struct.List* null

bb14:		; preds = %generate_patient.exit, %entry
	ret %struct.List* undef
}
