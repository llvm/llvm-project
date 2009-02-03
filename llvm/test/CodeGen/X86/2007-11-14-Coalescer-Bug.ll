; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=att | grep movl | count 1

	%struct.double_int = type { i64, i64 }
	%struct.tree_common = type <{ i8, [3 x i8] }>
	%struct.tree_int_cst = type { %struct.tree_common, %struct.double_int }
	%struct.tree_node = type { %struct.tree_int_cst }
@tree_code_type = external constant [0 x i32]		; <[0 x i32]*> [#uses=1]

define i32 @simple_cst_equal(%struct.tree_node* %t1, %struct.tree_node* %t2) {
entry:
	%tmp2526 = bitcast %struct.tree_node* %t1 to i32*		; <i32*> [#uses=1]
	br i1 false, label %UnifiedReturnBlock, label %bb21

bb21:		; preds = %entry
	%tmp27 = load i32* %tmp2526, align 4		; <i32> [#uses=1]
	%tmp29 = and i32 %tmp27, 255		; <i32> [#uses=3]
	%tmp2930 = trunc i32 %tmp29 to i8		; <i8> [#uses=1]
	%tmp37 = load i32* null, align 4		; <i32> [#uses=1]
	%tmp39 = and i32 %tmp37, 255		; <i32> [#uses=2]
	%tmp3940 = trunc i32 %tmp39 to i8		; <i8> [#uses=1]
	%tmp43 = add i32 %tmp29, -3		; <i32> [#uses=1]
	%tmp44 = icmp ult i32 %tmp43, 3		; <i1> [#uses=1]
	br i1 %tmp44, label %bb47.split, label %bb76

bb47.split:		; preds = %bb21
	ret i32 0

bb76:		; preds = %bb21
	br i1 false, label %bb82, label %bb146.split

bb82:		; preds = %bb76
	%tmp94 = getelementptr [0 x i32]* @tree_code_type, i32 0, i32 %tmp39		; <i32*> [#uses=1]
	%tmp95 = load i32* %tmp94, align 4		; <i32> [#uses=1]
	%tmp9596 = trunc i32 %tmp95 to i8		; <i8> [#uses=1]
	%tmp98 = add i8 %tmp9596, -4		; <i8> [#uses=1]
	%tmp99 = icmp ugt i8 %tmp98, 5		; <i1> [#uses=1]
	br i1 %tmp99, label %bb102, label %bb106

bb102:		; preds = %bb82
	ret i32 0

bb106:		; preds = %bb82
	ret i32 0

bb146.split:		; preds = %bb76
	%tmp149 = icmp eq i8 %tmp2930, %tmp3940		; <i1> [#uses=1]
	br i1 %tmp149, label %bb153, label %UnifiedReturnBlock

bb153:		; preds = %bb146.split
	switch i32 %tmp29, label %UnifiedReturnBlock [
		 i32 0, label %bb155
		 i32 1, label %bb187
	]

bb155:		; preds = %bb153
	ret i32 0

bb187:		; preds = %bb153
	%tmp198 = icmp eq %struct.tree_node* %t1, %t2		; <i1> [#uses=1]
	br i1 %tmp198, label %bb201, label %UnifiedReturnBlock

bb201:		; preds = %bb187
	ret i32 0

UnifiedReturnBlock:		; preds = %bb187, %bb153, %bb146.split, %entry
	ret i32 0
}
