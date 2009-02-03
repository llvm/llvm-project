; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin -mattr=+sse2 -disable-fp-elim | grep movss | count 1
; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin -mattr=+sse2 -disable-fp-elim -stats |& grep {Number of re-materialization} | grep 1

	%struct..0objc_object = type opaque
	%struct.OhBoy = type {  }
	%struct.BooHoo = type { i32 }
	%struct.objc_selector = type opaque
@llvm.used = appending global [1 x i8*] [ i8* bitcast (void (%struct.OhBoy*, %struct.objc_selector*, i32, %struct.BooHoo*)* @"-[MessageHeaderDisplay adjustFontSizeBy:viewingState:]" to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define void @"-[MessageHeaderDisplay adjustFontSizeBy:viewingState:]"(%struct.OhBoy* %self, %struct.objc_selector* %_cmd, i32 %delta, %struct.BooHoo* %viewingState) nounwind  {
entry:
	%tmp19 = load i32* null, align 4		; <i32> [#uses=1]
	%tmp24 = tail call float bitcast (void (%struct..0objc_object*, ...)* @objc_msgSend_fpret to float (%struct..0objc_object*, %struct.objc_selector*)*)( %struct..0objc_object* null, %struct.objc_selector* null ) nounwind 		; <float> [#uses=2]
	%tmp30 = icmp sgt i32 %delta, 0		; <i1> [#uses=1]
	br i1 %tmp30, label %bb33, label %bb87.preheader
bb33:		; preds = %entry
	%tmp28 = add float 0.000000e+00, %tmp24		; <float> [#uses=1]
	%tmp35 = fcmp ogt float %tmp28, 1.800000e+01		; <i1> [#uses=1]
	br i1 %tmp35, label %bb38, label %bb87.preheader
bb38:		; preds = %bb33
	%tmp53 = add i32 %tmp19, %delta		; <i32> [#uses=2]
	br i1 false, label %bb50, label %bb43
bb43:		; preds = %bb38
	store i32 %tmp53, i32* null, align 4
	ret void
bb50:		; preds = %bb38
	%tmp56 = sub float 1.800000e+01, %tmp24		; <float> [#uses=1]
	%tmp57 = fcmp ugt float 0.000000e+00, %tmp56		; <i1> [#uses=1]
	br i1 %tmp57, label %bb64, label %bb87.preheader
bb64:		; preds = %bb50
	ret void
bb87.preheader:		; preds = %bb50, %bb33, %entry
	%usableDelta.0 = phi i32 [ %delta, %entry ], [ %delta, %bb33 ], [ %tmp53, %bb50 ]		; <i32> [#uses=1]
	%tmp100 = tail call %struct..0objc_object* (%struct..0objc_object*, %struct.objc_selector*, ...)* @objc_msgSend( %struct..0objc_object* null, %struct.objc_selector* null, %struct..0objc_object* null ) nounwind 		; <%struct..0objc_object*> [#uses=2]
	%tmp106 = tail call %struct..0objc_object* (%struct..0objc_object*, %struct.objc_selector*, ...)* @objc_msgSend( %struct..0objc_object* %tmp100, %struct.objc_selector* null ) nounwind 		; <%struct..0objc_object*> [#uses=0]
	%umax = select i1 false, i32 1, i32 0		; <i32> [#uses=1]
	br label %bb108
bb108:		; preds = %bb108, %bb87.preheader
	%attachmentIndex.0.reg2mem.0 = phi i32 [ 0, %bb87.preheader ], [ %indvar.next, %bb108 ]		; <i32> [#uses=2]
	%tmp114 = tail call %struct..0objc_object* (%struct..0objc_object*, %struct.objc_selector*, ...)* @objc_msgSend( %struct..0objc_object* %tmp100, %struct.objc_selector* null, i32 %attachmentIndex.0.reg2mem.0 ) nounwind 		; <%struct..0objc_object*> [#uses=1]
	%tmp121 = tail call %struct..0objc_object* (%struct..0objc_object*, %struct.objc_selector*, ...)* @objc_msgSend( %struct..0objc_object* %tmp114, %struct.objc_selector* null, i32 %usableDelta.0 ) nounwind 		; <%struct..0objc_object*> [#uses=0]
	%indvar.next = add i32 %attachmentIndex.0.reg2mem.0, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %umax		; <i1> [#uses=1]
	br i1 %exitcond, label %bb130, label %bb108
bb130:		; preds = %bb108
	ret void
}

declare %struct..0objc_object* @objc_msgSend(%struct..0objc_object*, %struct.objc_selector*, ...)

declare void @objc_msgSend_fpret(%struct..0objc_object*, ...)
