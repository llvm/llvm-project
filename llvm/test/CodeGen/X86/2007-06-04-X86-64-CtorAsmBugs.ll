; RUN: llc < %s -mtriple=x86_64-apple-darwin | not grep GOTPCREL
; RUN: llc < %s -mtriple=x86_64-apple-darwin | grep ".align.*3"

	%struct.A = type { [1024 x i8] }
@_ZN1A1aE = global %struct.A zeroinitializer, align 32		; <ptr> [#uses=1]
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [ { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__I__ZN1A1aE, ptr null } ]		; <ptr> [#uses=0]

define internal void @_GLOBAL__I__ZN1A1aE() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
	br label %bb.i

bb.i:		; preds = %bb.i, %entry
	%i.1.i1.0 = phi i32 [ 0, %entry ], [ %indvar.next, %bb.i ]		; <i32> [#uses=2]
	%tmp1012.i = sext i32 %i.1.i1.0 to i64		; <i64> [#uses=1]
	%tmp13.i = getelementptr %struct.A, ptr @_ZN1A1aE, i32 0, i32 0, i64 %tmp1012.i		; <ptr> [#uses=1]
	store i8 0, ptr %tmp13.i
	%indvar.next = add i32 %i.1.i1.0, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, 1024		; <i1> [#uses=1]
	br i1 %exitcond, label %_Z41__static_initialization_and_destruction_0ii.exit, label %bb.i

_Z41__static_initialization_and_destruction_0ii.exit:		; preds = %bb.i
	ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
	ret i32 0
}
