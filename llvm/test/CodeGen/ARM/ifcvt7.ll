; RUN: llc < %s -mtriple=armv7-apple-darwin | FileCheck %s
; FIXME: Need post-ifcvt branch folding to get rid of the extra br at end of BB1.

	%struct.quad_struct = type { i32, i32, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct* }

define fastcc i32 @CountTree(%struct.quad_struct* %tree) {
; CHECK: cmpeq
; CHECK: moveq
; CHECK: popeq
entry:
	br label %tailrecurse

tailrecurse:		; preds = %bb, %entry
	%tmp6 = load %struct.quad_struct** null		; <%struct.quad_struct*> [#uses=1]
	%tmp9 = load %struct.quad_struct** null		; <%struct.quad_struct*> [#uses=2]
	%tmp12 = load %struct.quad_struct** null		; <%struct.quad_struct*> [#uses=1]
	%tmp14 = icmp eq %struct.quad_struct* null, null		; <i1> [#uses=1]
	%tmp17 = icmp eq %struct.quad_struct* %tmp6, null		; <i1> [#uses=1]
	%tmp23 = icmp eq %struct.quad_struct* %tmp9, null		; <i1> [#uses=1]
	%tmp29 = icmp eq %struct.quad_struct* %tmp12, null		; <i1> [#uses=1]
	%bothcond = and i1 %tmp17, %tmp14		; <i1> [#uses=1]
	%bothcond1 = and i1 %bothcond, %tmp23		; <i1> [#uses=1]
	%bothcond2 = and i1 %bothcond1, %tmp29		; <i1> [#uses=1]
	br i1 %bothcond2, label %return, label %bb

bb:		; preds = %tailrecurse
	%tmp41 = tail call fastcc i32 @CountTree( %struct.quad_struct* %tmp9 )		; <i32> [#uses=0]
	br label %tailrecurse

return:		; preds = %tailrecurse
	ret i32 0
}
