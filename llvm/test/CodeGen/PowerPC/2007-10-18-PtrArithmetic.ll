; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -mattr=+altivec
	%struct.inoutprops = type <{ i8, [3 x i8] }>

define void @bork(ptr %argA, ptr %argB, ptr %res, i8 %inoutspec.0) {
entry:
	%.mask = and i8 %inoutspec.0, -16		; <i8> [#uses=1]
	%tmp6 = icmp eq i8 %.mask, 16		; <i1> [#uses=1]
	br i1 %tmp6, label %cond_true, label %UnifiedReturnBlock

cond_true:		; preds = %entry
	%tmp14 = load <4 x i32>, ptr %argA, align 16		; <<4 x i32>> [#uses=1]
	%tmp18 = load <4 x i32>, ptr %argB, align 16		; <<4 x i32>> [#uses=1]
	%tmp19 = sdiv <4 x i32> %tmp14, %tmp18		; <<4 x i32>> [#uses=1]
	store <4 x i32> %tmp19, ptr %res, align 16
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}
