; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin -mattr=+sse2 | grep mov | count 6

	%struct.quad_struct = type { i32, i32, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct* }

define  i32 @perimeter(%struct.quad_struct* %tree, i32 %size) nounwind  {
entry:
	switch i32 %size, label %UnifiedReturnBlock [
		 i32 2, label %bb
		 i32 0, label %bb50
	]

bb:		; preds = %entry
	%tmp31 = tail call  i32 @perimeter( %struct.quad_struct* null, i32 0 ) nounwind 		; <i32> [#uses=1]
	%tmp40 = tail call  i32 @perimeter( %struct.quad_struct* null, i32 0 ) nounwind 		; <i32> [#uses=1]
	%tmp33 = add i32 0, %tmp31		; <i32> [#uses=1]
	%tmp42 = add i32 %tmp33, %tmp40		; <i32> [#uses=1]
	ret i32 %tmp42

bb50:		; preds = %entry
	ret i32 0

UnifiedReturnBlock:		; preds = %entry
	ret i32 0
}
