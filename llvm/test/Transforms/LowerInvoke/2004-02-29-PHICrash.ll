; RUN: llvm-as < %s | opt -lowerinvoke -enable-correct-eh-support -disable-output

define void @_ZNKSt11__use_cacheISt16__numpunct_cacheIcEEclERKSt6locale() {
entry:
	br i1 false, label %then, label %UnifiedReturnBlock
then:		; preds = %entry
	invoke void @_Znwj( )
			to label %UnifiedReturnBlock unwind label %UnifiedReturnBlock
UnifiedReturnBlock:		; preds = %then, %then, %entry
	%UnifiedRetVal = phi i32* [ null, %entry ], [ null, %then ], [ null, %then ] ; <i32*> [#uses=0]
	ret void
}

declare void @_Znwj()

