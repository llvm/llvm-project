; RUN: llc < %s -mtriple=i686-- | not grep IMPLICIT_DEF

	%struct.node_t = type { double*, %struct.node_t*, %struct.node_t**, double**, double*, i32, i32 }

define void @localize_local_bb19_bb(%struct.node_t** %cur_node) {
newFuncRoot:
	%tmp1 = load %struct.node_t*, %struct.node_t** %cur_node, align 4		; <%struct.node_t*> [#uses=1]
	%tmp2 = getelementptr %struct.node_t, %struct.node_t* %tmp1, i32 0, i32 4		; <double**> [#uses=1]
	%tmp3 = load double*, double** %tmp2, align 4		; <double*> [#uses=1]
	%tmp4 = load %struct.node_t*, %struct.node_t** %cur_node, align 4		; <%struct.node_t*> [#uses=1]
	%tmp5 = getelementptr %struct.node_t, %struct.node_t* %tmp4, i32 0, i32 4		; <double**> [#uses=1]
	store double* %tmp3, double** %tmp5, align 4
	%tmp6 = load %struct.node_t*, %struct.node_t** %cur_node, align 4		; <%struct.node_t*> [#uses=1]
	%tmp7 = getelementptr %struct.node_t, %struct.node_t* %tmp6, i32 0, i32 3		; <double***> [#uses=1]
	%tmp8 = load double**, double*** %tmp7, align 4		; <double**> [#uses=1]
	%tmp9 = load %struct.node_t*, %struct.node_t** %cur_node, align 4		; <%struct.node_t*> [#uses=1]
	%tmp10 = getelementptr %struct.node_t, %struct.node_t* %tmp9, i32 0, i32 3		; <double***> [#uses=1]
	store double** %tmp8, double*** %tmp10, align 4
	%tmp11 = load %struct.node_t*, %struct.node_t** %cur_node, align 4		; <%struct.node_t*> [#uses=1]
	%tmp12 = getelementptr %struct.node_t, %struct.node_t* %tmp11, i32 0, i32 0		; <double**> [#uses=1]
	%tmp13 = load double*, double** %tmp12, align 4		; <double*> [#uses=1]
	%tmp14 = load %struct.node_t*, %struct.node_t** %cur_node, align 4		; <%struct.node_t*> [#uses=1]
	%tmp15 = getelementptr %struct.node_t, %struct.node_t* %tmp14, i32 0, i32 0		; <double**> [#uses=1]
	store double* %tmp13, double** %tmp15, align 4
	%tmp16 = load %struct.node_t*, %struct.node_t** %cur_node, align 4		; <%struct.node_t*> [#uses=1]
	%tmp17 = getelementptr %struct.node_t, %struct.node_t* %tmp16, i32 0, i32 1		; <%struct.node_t**> [#uses=1]
	%tmp18 = load %struct.node_t*, %struct.node_t** %tmp17, align 4		; <%struct.node_t*> [#uses=1]
	store %struct.node_t* %tmp18, %struct.node_t** %cur_node, align 4
	ret void
}
