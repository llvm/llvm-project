; RUN: llc < %s -mtriple=thumbv7-apple-ios -arm-atomic-cfg-tidy=0 | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-ios -arm-atomic-cfg-tidy=0 -arm-default-it | FileCheck %s
; RUN: llc < %s -mtriple=thumbv8-apple-ios -arm-atomic-cfg-tidy=0 -arm-no-restrict-it | FileCheck %s

define void @foo(i32 %X, i32 %Y) {
entry:
; CHECK-LABEL: foo:
; CHECK: it ne
; CHECK: cmpne
; CHECK: it hi
; CHECK: pophi {r7, pc}
	%tmp1 = icmp ult i32 %X, 4		; <i1> [#uses=1]
	%tmp4 = icmp eq i32 %Y, 0		; <i1> [#uses=1]
	%tmp7 = or i1 %tmp4, %tmp1		; <i1> [#uses=1]
	br i1 %tmp7, label %cond_true, label %UnifiedReturnBlock

cond_true:		; preds = %entry
	%tmp10 = call i32 (...) @bar( )		; <i32> [#uses=0]
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

declare i32 @bar(...)

; FIXME: Need post-ifcvt branch folding to get rid of the extra br at end of BB1.

	%struct.quad_struct = type { i32, i32, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct* }

define fastcc i32 @CountTree(%struct.quad_struct* %tree) {
entry:
; CHECK-LABEL: CountTree:
; CHECK: bne
; CHECK: cmp
; CHECK: it eq
; CHECK: cmpeq
; CHECK: itt eq
; CHECK: moveq
; CHECK: popeq
	br label %tailrecurse

tailrecurse:		; preds = %bb, %entry
	%tmp6 = load %struct.quad_struct*, %struct.quad_struct** null		; <%struct.quad_struct*> [#uses=1]
	%tmp9 = load %struct.quad_struct*, %struct.quad_struct** null		; <%struct.quad_struct*> [#uses=2]
	%tmp12 = load %struct.quad_struct*, %struct.quad_struct** null		; <%struct.quad_struct*> [#uses=1]
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

	%struct.SString = type { i8*, i32, i32 }

declare void @abort()

define fastcc void @t1(%struct.SString* %word, i8 signext  %c) {
entry:
; CHECK-LABEL: t1:
; CHECK: it ne
; CHECK: popne {r7, pc}
	%tmp1 = icmp eq %struct.SString* %word, null		; <i1> [#uses=1]
	br i1 %tmp1, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	tail call void @abort( )
	unreachable

cond_false:		; preds = %entry
	ret void
}

define fastcc void @t2() nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK: cmp r0, #0
; CHECK: %growMapping.exit
	br i1 undef, label %bb.i.i3, label %growMapping.exit

bb.i.i3:		; preds = %entry
	unreachable

growMapping.exit:		; preds = %entry
	unreachable
}
