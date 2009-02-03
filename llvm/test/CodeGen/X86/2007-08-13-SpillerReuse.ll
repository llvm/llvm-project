; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin | grep "48(%esp)" | count 5

	%struct..0anon = type { i32 }
	%struct.rtvec_def = type { i32, [1 x %struct..0anon] }
	%struct.rtx_def = type { i16, i8, i8, [1 x %struct..0anon] }
@rtx_format = external global [116 x i8*]		; <[116 x i8*]*> [#uses=1]
@rtx_length = external global [117 x i32]		; <[117 x i32]*> [#uses=1]

declare %struct.rtx_def* @fixup_memory_subreg(%struct.rtx_def*, %struct.rtx_def*, i32)

define %struct.rtx_def* @walk_fixup_memory_subreg(%struct.rtx_def* %x, %struct.rtx_def* %insn) {
entry:
	%tmp2 = icmp eq %struct.rtx_def* %x, null		; <i1> [#uses=1]
	br i1 %tmp2, label %UnifiedReturnBlock, label %cond_next

cond_next:		; preds = %entry
	%tmp6 = getelementptr %struct.rtx_def* %x, i32 0, i32 0		; <i16*> [#uses=1]
	%tmp7 = load i16* %tmp6		; <i16> [#uses=2]
	%tmp78 = zext i16 %tmp7 to i32		; <i32> [#uses=2]
	%tmp10 = icmp eq i16 %tmp7, 54		; <i1> [#uses=1]
	br i1 %tmp10, label %cond_true13, label %cond_next32

cond_true13:		; preds = %cond_next
	%tmp15 = getelementptr %struct.rtx_def* %x, i32 0, i32 3		; <[1 x %struct..0anon]*> [#uses=1]
	%tmp1718 = bitcast [1 x %struct..0anon]* %tmp15 to %struct.rtx_def**		; <%struct.rtx_def**> [#uses=1]
	%tmp19 = load %struct.rtx_def** %tmp1718		; <%struct.rtx_def*> [#uses=1]
	%tmp20 = getelementptr %struct.rtx_def* %tmp19, i32 0, i32 0		; <i16*> [#uses=1]
	%tmp21 = load i16* %tmp20		; <i16> [#uses=1]
	%tmp22 = icmp eq i16 %tmp21, 57		; <i1> [#uses=1]
	br i1 %tmp22, label %cond_true25, label %cond_next32

cond_true25:		; preds = %cond_true13
	%tmp29 = tail call %struct.rtx_def* @fixup_memory_subreg( %struct.rtx_def* %x, %struct.rtx_def* %insn, i32 1 )		; <%struct.rtx_def*> [#uses=1]
	ret %struct.rtx_def* %tmp29

cond_next32:		; preds = %cond_true13, %cond_next
	%tmp34 = getelementptr [116 x i8*]* @rtx_format, i32 0, i32 %tmp78		; <i8**> [#uses=1]
	%tmp35 = load i8** %tmp34, align 4		; <i8*> [#uses=1]
	%tmp37 = getelementptr [117 x i32]* @rtx_length, i32 0, i32 %tmp78		; <i32*> [#uses=1]
	%tmp38 = load i32* %tmp37, align 4		; <i32> [#uses=1]
	%i.011 = add i32 %tmp38, -1		; <i32> [#uses=2]
	%tmp12513 = icmp sgt i32 %i.011, -1		; <i1> [#uses=1]
	br i1 %tmp12513, label %bb, label %UnifiedReturnBlock

bb:		; preds = %bb123, %cond_next32
	%indvar = phi i32 [ %indvar.next26, %bb123 ], [ 0, %cond_next32 ]		; <i32> [#uses=2]
	%i.01.0 = sub i32 %i.011, %indvar		; <i32> [#uses=5]
	%tmp42 = getelementptr i8* %tmp35, i32 %i.01.0		; <i8*> [#uses=2]
	%tmp43 = load i8* %tmp42		; <i8> [#uses=1]
	switch i8 %tmp43, label %bb123 [
		 i8 101, label %cond_true47
		 i8 69, label %bb105.preheader
	]

cond_true47:		; preds = %bb
	%tmp52 = getelementptr %struct.rtx_def* %x, i32 0, i32 3, i32 %i.01.0		; <%struct..0anon*> [#uses=1]
	%tmp5354 = bitcast %struct..0anon* %tmp52 to %struct.rtx_def**		; <%struct.rtx_def**> [#uses=1]
	%tmp55 = load %struct.rtx_def** %tmp5354		; <%struct.rtx_def*> [#uses=1]
	%tmp58 = tail call  %struct.rtx_def* @walk_fixup_memory_subreg( %struct.rtx_def* %tmp55, %struct.rtx_def* %insn )		; <%struct.rtx_def*> [#uses=1]
	%tmp62 = getelementptr %struct.rtx_def* %x, i32 0, i32 3, i32 %i.01.0, i32 0		; <i32*> [#uses=1]
	%tmp58.c = ptrtoint %struct.rtx_def* %tmp58 to i32		; <i32> [#uses=1]
	store i32 %tmp58.c, i32* %tmp62
	%tmp6816 = load i8* %tmp42		; <i8> [#uses=1]
	%tmp6917 = icmp eq i8 %tmp6816, 69		; <i1> [#uses=1]
	br i1 %tmp6917, label %bb105.preheader, label %bb123

bb105.preheader:		; preds = %cond_true47, %bb
	%tmp11020 = getelementptr %struct.rtx_def* %x, i32 0, i32 3, i32 %i.01.0		; <%struct..0anon*> [#uses=1]
	%tmp11111221 = bitcast %struct..0anon* %tmp11020 to %struct.rtvec_def**		; <%struct.rtvec_def**> [#uses=3]
	%tmp11322 = load %struct.rtvec_def** %tmp11111221		; <%struct.rtvec_def*> [#uses=1]
	%tmp11423 = getelementptr %struct.rtvec_def* %tmp11322, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp11524 = load i32* %tmp11423		; <i32> [#uses=1]
	%tmp11625 = icmp eq i32 %tmp11524, 0		; <i1> [#uses=1]
	br i1 %tmp11625, label %bb123, label %bb73

bb73:		; preds = %bb73, %bb105.preheader
	%j.019 = phi i32 [ %tmp104, %bb73 ], [ 0, %bb105.preheader ]		; <i32> [#uses=3]
	%tmp81 = load %struct.rtvec_def** %tmp11111221		; <%struct.rtvec_def*> [#uses=2]
	%tmp92 = getelementptr %struct.rtvec_def* %tmp81, i32 0, i32 1, i32 %j.019		; <%struct..0anon*> [#uses=1]
	%tmp9394 = bitcast %struct..0anon* %tmp92 to %struct.rtx_def**		; <%struct.rtx_def**> [#uses=1]
	%tmp95 = load %struct.rtx_def** %tmp9394		; <%struct.rtx_def*> [#uses=1]
	%tmp98 = tail call  %struct.rtx_def* @walk_fixup_memory_subreg( %struct.rtx_def* %tmp95, %struct.rtx_def* %insn )		; <%struct.rtx_def*> [#uses=1]
	%tmp101 = getelementptr %struct.rtvec_def* %tmp81, i32 0, i32 1, i32 %j.019, i32 0		; <i32*> [#uses=1]
	%tmp98.c = ptrtoint %struct.rtx_def* %tmp98 to i32		; <i32> [#uses=1]
	store i32 %tmp98.c, i32* %tmp101
	%tmp104 = add i32 %j.019, 1		; <i32> [#uses=2]
	%tmp113 = load %struct.rtvec_def** %tmp11111221		; <%struct.rtvec_def*> [#uses=1]
	%tmp114 = getelementptr %struct.rtvec_def* %tmp113, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp115 = load i32* %tmp114		; <i32> [#uses=1]
	%tmp116 = icmp ult i32 %tmp104, %tmp115		; <i1> [#uses=1]
	br i1 %tmp116, label %bb73, label %bb123

bb123:		; preds = %bb73, %bb105.preheader, %cond_true47, %bb
	%i.0 = add i32 %i.01.0, -1		; <i32> [#uses=1]
	%tmp125 = icmp sgt i32 %i.0, -1		; <i1> [#uses=1]
	%indvar.next26 = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp125, label %bb, label %UnifiedReturnBlock

UnifiedReturnBlock:		; preds = %bb123, %cond_next32, %entry
	%UnifiedRetVal = phi %struct.rtx_def* [ null, %entry ], [ %x, %cond_next32 ], [ %x, %bb123 ]		; <%struct.rtx_def*> [#uses=1]
	ret %struct.rtx_def* %UnifiedRetVal
}
