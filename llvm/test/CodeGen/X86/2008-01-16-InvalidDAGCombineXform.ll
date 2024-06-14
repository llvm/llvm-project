; RUN: llc < %s -mtriple=i686-- | not grep IMPLICIT_DEF

	%struct.node_t = type { ptr, ptr, ptr, ptr, ptr, i32, i32 }

define void @localize_local_bb19_bb(ptr %cur_node) {
newFuncRoot:
	%tmp1 = load ptr, ptr %cur_node, align 4		; <ptr> [#uses=1]
	%tmp2 = getelementptr %struct.node_t, ptr %tmp1, i32 0, i32 4		; <ptr> [#uses=1]
	%tmp3 = load ptr, ptr %tmp2, align 4		; <ptr> [#uses=1]
	%tmp4 = load ptr, ptr %cur_node, align 4		; <ptr> [#uses=1]
	%tmp5 = getelementptr %struct.node_t, ptr %tmp4, i32 0, i32 4		; <ptr> [#uses=1]
	store ptr %tmp3, ptr %tmp5, align 4
	%tmp6 = load ptr, ptr %cur_node, align 4		; <ptr> [#uses=1]
	%tmp7 = getelementptr %struct.node_t, ptr %tmp6, i32 0, i32 3		; <ptr> [#uses=1]
	%tmp8 = load ptr, ptr %tmp7, align 4		; <ptr> [#uses=1]
	%tmp9 = load ptr, ptr %cur_node, align 4		; <ptr> [#uses=1]
	%tmp10 = getelementptr %struct.node_t, ptr %tmp9, i32 0, i32 3		; <ptr> [#uses=1]
	store ptr %tmp8, ptr %tmp10, align 4
	%tmp11 = load ptr, ptr %cur_node, align 4		; <ptr> [#uses=1]
	%tmp12 = getelementptr %struct.node_t, ptr %tmp11, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp13 = load ptr, ptr %tmp12, align 4		; <ptr> [#uses=1]
	%tmp14 = load ptr, ptr %cur_node, align 4		; <ptr> [#uses=1]
	%tmp15 = getelementptr %struct.node_t, ptr %tmp14, i32 0, i32 0		; <ptr> [#uses=1]
	store ptr %tmp13, ptr %tmp15, align 4
	%tmp16 = load ptr, ptr %cur_node, align 4		; <ptr> [#uses=1]
	%tmp17 = getelementptr %struct.node_t, ptr %tmp16, i32 0, i32 1		; <ptr> [#uses=1]
	%tmp18 = load ptr, ptr %tmp17, align 4		; <ptr> [#uses=1]
	store ptr %tmp18, ptr %cur_node, align 4
	ret void
}
