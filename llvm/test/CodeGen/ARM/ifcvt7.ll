; RUN: llc < %s -mtriple=armv7-apple-darwin | FileCheck %s
; FIXME: Need post-ifcvt branch folding to get rid of the extra br at end of BB1.

	%struct.quad_struct = type { i32, i32, ptr, ptr, ptr, ptr, ptr }

define fastcc i32 @CountTree(ptr %tree) {
; CHECK: cmpeq
entry:
	br label %tailrecurse

tailrecurse:		; preds = %bb, %entry
	%tmp6 = load ptr, ptr null		; <ptr> [#uses=1]
	%tmp9 = load ptr, ptr null		; <ptr> [#uses=2]
	%tmp12 = load ptr, ptr null		; <ptr> [#uses=1]
	%tmp14 = icmp eq ptr null, null		; <i1> [#uses=1]
	%tmp17 = icmp eq ptr %tmp6, null		; <i1> [#uses=1]
	%tmp23 = icmp eq ptr %tmp9, null		; <i1> [#uses=1]
	%tmp29 = icmp eq ptr %tmp12, null		; <i1> [#uses=1]
	%bothcond = and i1 %tmp17, %tmp14		; <i1> [#uses=1]
	%bothcond1 = and i1 %bothcond, %tmp23		; <i1> [#uses=1]
	%bothcond2 = and i1 %bothcond1, %tmp29		; <i1> [#uses=1]
	br i1 %bothcond2, label %return, label %bb

bb:		; preds = %tailrecurse
	%tmp41 = tail call fastcc i32 @CountTree( ptr %tmp9 )		; <i32> [#uses=0]
	br label %tailrecurse

return:		; preds = %tailrecurse
	ret i32 0
}
