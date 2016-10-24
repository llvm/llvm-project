; RUN: llc < %s -mtriple=thumbv7-apple-darwin -relocation-model=pic | FileCheck %s

; Thumb2 target should reorder the bb's in order to use tbb / tbh.

	%struct.R_flstr = type { i32, i32, i8* }
	%struct._T_tstr = type { i32, %struct.R_flstr*, %struct._T_tstr* }
@_C_nextcmd = external global i32		; <i32*> [#uses=3]
@.str31 = external constant [28 x i8], align 1		; <[28 x i8]*> [#uses=1]
@_T_gtol = external global %struct._T_tstr*		; <%struct._T_tstr**> [#uses=2]

declare i32 @strlen(i8* nocapture) nounwind readonly

declare void @Z_fatal(i8*) noreturn nounwind

declare noalias i8* @calloc(i32, i32) nounwind

; Jump tables are not anchored next to the TBB/TBH any more. Make sure the
; correct address is still calculated (i.e. via a PC-relative symbol *at* the
; TBB/TBH).
define i32 @main(i32 %argc, i8** nocapture %argv) nounwind {
; CHECK-LABEL: main:
; CHECK-NOT: adr {{r[0-9]+}}, LJTI
; CHECK: [[PCREL_ANCHOR:LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT:     tbb [pc, {{r[0-9]+}}]

; CHECK: LJTI0_0:
; CHECK-NEXT: .data_region jt8
; CHECK-NEXT: .byte (LBB{{[0-9]+_[0-9]+}}-([[PCREL_ANCHOR]]+4))/2

entry:
	br label %bb42.i

bb1.i2:		; preds = %bb42.i
	br label %bb40.i

bb5.i:		; preds = %bb42.i
	%0 = or i32 %argc, 32		; <i32> [#uses=1]
	br label %bb40.i

bb7.i:		; preds = %bb42.i
	call  void @_T_addtol(%struct._T_tstr** @_T_gtol, i32 0, i8* null) nounwind
	unreachable

bb15.i:		; preds = %bb42.i
	call  void @_T_addtol(%struct._T_tstr** @_T_gtol, i32 2, i8* null) nounwind
	unreachable

bb23.i:		; preds = %bb42.i
	%1 = call  i32 @strlen(i8* null) nounwind readonly		; <i32> [#uses=0]
	unreachable

bb33.i:		; preds = %bb42.i
	store i32 0, i32* @_C_nextcmd, align 4
	%2 = call  noalias i8* @calloc(i32 21, i32 1) nounwind		; <i8*> [#uses=0]
	unreachable

bb34.i:		; preds = %bb42.i
	%3 = load i32, i32* @_C_nextcmd, align 4		; <i32> [#uses=1]
	%4 = add i32 %3, 1		; <i32> [#uses=1]
	store i32 %4, i32* @_C_nextcmd, align 4
	%5 = call  noalias i8* @calloc(i32 22, i32 1) nounwind		; <i8*> [#uses=0]
	unreachable

bb35.i:		; preds = %bb42.i
	%6 = call  noalias i8* @calloc(i32 20, i32 1) nounwind		; <i8*> [#uses=0]
	unreachable

bb37.i:		; preds = %bb42.i
	%7 = call  noalias i8* @calloc(i32 14, i32 1) nounwind		; <i8*> [#uses=0]
	unreachable

bb39.i:		; preds = %bb42.i
	call  void @Z_fatal(i8* getelementptr ([28 x i8], [28 x i8]* @.str31, i32 0, i32 0)) nounwind
	unreachable

bb40.i:		; preds = %bb42.i, %bb5.i, %bb1.i2
	br label %bb42.i

bb42.i:		; preds = %bb40.i, %entry
	switch i32 %argc, label %bb39.i [
		i32 67, label %bb33.i
		i32 70, label %bb35.i
		i32 77, label %bb37.i
		i32 83, label %bb34.i
		i32 97, label %bb7.i
		i32 100, label %bb5.i
		i32 101, label %bb40.i
		i32 102, label %bb23.i
		i32 105, label %bb15.i
		i32 116, label %bb1.i2
	]
}

declare void @_T_addtol(%struct._T_tstr** nocapture, i32, i8*) nounwind
