; RUN: llc < %s -mtriple=i686-pc-linux-gnu -mcpu=athlon -relocation-model=pic
; PR1545

@.str97 = external constant [56 x i8]		; <ptr> [#uses=1]

declare void @PR_LogPrint(ptr, ...)

define i32 @_ZN13nsPrintEngine19SetupToPrintContentEP16nsIDeviceContextP12nsIDOMWindow() {
entry:
	br i1 false, label %cond_true122, label %cond_next453

cond_true122:		; preds = %entry
	br i1 false, label %bb164, label %cond_true136

cond_true136:		; preds = %cond_true122
	ret i32 0

bb164:		; preds = %cond_true122
	br i1 false, label %bb383, label %cond_true354

cond_true354:		; preds = %bb164
	ret i32 0

bb383:		; preds = %bb164
	%tmp408 = load float, ptr null		; <float> [#uses=2]
	br i1 false, label %cond_true425, label %cond_next443

cond_true425:		; preds = %bb383
	%tmp430 = load float, ptr null		; <float> [#uses=1]
	%tmp432 = fsub float %tmp430, %tmp408		; <float> [#uses=1]
	%tmp432433 = fpext float %tmp432 to double		; <double> [#uses=1]
	%tmp434435 = fpext float %tmp408 to double		; <double> [#uses=1]
	call void (ptr, ...) @PR_LogPrint( ptr @.str97, double 0.000000e+00, double %tmp434435, double %tmp432433 )
	ret i32 0

cond_next443:		; preds = %bb383
	ret i32 0

cond_next453:		; preds = %entry
	ret i32 0
}
