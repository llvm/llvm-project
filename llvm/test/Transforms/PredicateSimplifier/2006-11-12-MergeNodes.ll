; RUN: llvm-as < %s | opt -predsimplify -disable-output
; END.
target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"
deplibs = [ "c", "crtend" ]
	%struct.VDIR_ST = type { i32, i32, i32, %struct.acl*, %struct.pfile*, %struct.vlink*, %struct.vlink*, %struct.vlink*, %struct.VDIR_ST*, %struct.VDIR_ST* }
	%struct.acl = type { i32, i8*, i8*, i8*, %struct.restrict*, %struct.acl*, %struct.acl* }
	%struct.avalue = type { i8* }
	%struct.pattrib = type { i8, i8*, i8*, %struct.avalue, %struct.pattrib*, %struct.pattrib* }
	%struct.pfile = type { i32, i32, i32, i32, i32, %struct.vlink*, %struct.vlink*, %struct.pattrib*, %struct.pfile*, %struct.pfile* }
	%struct.restrict = type { %struct.acl*, %struct.acl* }
	%struct.vlink = type { i32, i8*, i8, i32, i8*, %struct.vlink*, %struct.vlink*, i8*, i8*, i8*, i8*, i32, i32, %struct.acl*, i32, i32, i8*, %struct.pattrib*, %struct.pfile*, %struct.vlink*, %struct.vlink* }

define void @vl_insert(%struct.vlink* %vl) {
entry:
	%tmp91 = call i32 @vl_comp( )		; <i32> [#uses=2]
	%tmp93 = icmp sgt i32 %tmp91, 0		; <i1> [#uses=1]
	br i1 %tmp93, label %cond_next84, label %bb94
cond_next84:		; preds = %entry
	ret void
bb94:		; preds = %entry
	%tmp96 = icmp eq i32 %tmp91, 0		; <i1> [#uses=1]
	br i1 %tmp96, label %cond_true97, label %cond_next203
cond_true97:		; preds = %bb94
	br i1 false, label %cond_next105, label %cond_true102
cond_true102:		; preds = %cond_true97
	ret void
cond_next105:		; preds = %cond_true97
	%tmp110 = getelementptr %struct.vlink* %vl, i32 0, i32 12		; <i32*> [#uses=1]
	%tmp111 = load i32* %tmp110		; <i32> [#uses=1]
	%tmp129 = icmp eq i32 %tmp111, 0		; <i1> [#uses=1]
	br i1 %tmp129, label %cond_true130, label %cond_next133
cond_true130:		; preds = %cond_next105
	ret void
cond_next133:		; preds = %cond_next105
	ret void
cond_next203:		; preds = %bb94
	ret void
}

declare i32 @vl_comp()
