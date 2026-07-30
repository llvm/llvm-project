; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--
; END.

	%struct.attr_desc = type { ptr, ptr, ptr, ptr, i32 }
	%struct.attr_value = type { ptr, ptr, ptr, i32, i32 }
	%struct.insn_def = type { ptr, ptr, i32, i32, i32, i32, i32 }
	%struct.insn_ent = type { ptr, ptr }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.u = type { [1 x i64] }

define void @find_attr() {
entry:
	%tmp26 = icmp eq ptr null, null		; <i1> [#uses=1]
	br i1 %tmp26, label %bb30, label %cond_true27
cond_true27:		; preds = %entry
	ret void
bb30:		; preds = %entry
	%tmp67 = icmp eq ptr null, null		; <i1> [#uses=1]
	br i1 %tmp67, label %cond_next92, label %cond_true68
cond_true68:		; preds = %bb30
	ret void
cond_next92:		; preds = %bb30
	%tmp173 = getelementptr %struct.attr_desc, ptr null, i32 0, i32 4		; <ptr> [#uses=2]
	%tmp174 = load i32, ptr %tmp173		; <i32> [#uses=1]
	%tmp177 = and i32 %tmp174, -9		; <i32> [#uses=1]
	store i32 %tmp177, ptr %tmp173
	%tmp180 = getelementptr %struct.attr_desc, ptr null, i32 0, i32 4		; <ptr> [#uses=1]
	%tmp181 = load i32, ptr %tmp180		; <i32> [#uses=1]
	%tmp185 = getelementptr %struct.attr_desc, ptr null, i32 0, i32 4		; <ptr> [#uses=2]
	%tmp186 = load i32, ptr %tmp185		; <i32> [#uses=1]
	%tmp183187 = shl i32 %tmp181, 1		; <i32> [#uses=1]
	%tmp188 = and i32 %tmp183187, 16		; <i32> [#uses=1]
	%tmp190 = and i32 %tmp186, -17		; <i32> [#uses=1]
	%tmp191 = or i32 %tmp190, %tmp188		; <i32> [#uses=1]
	store i32 %tmp191, ptr %tmp185
	%tmp193 = getelementptr %struct.attr_desc, ptr null, i32 0, i32 4		; <ptr> [#uses=1]
	%tmp194 = load i32, ptr %tmp193		; <i32> [#uses=1]
	%tmp198 = getelementptr %struct.attr_desc, ptr null, i32 0, i32 4		; <ptr> [#uses=2]
	%tmp199 = load i32, ptr %tmp198		; <i32> [#uses=1]
	%tmp196200 = shl i32 %tmp194, 2		; <i32> [#uses=1]
	%tmp201 = and i32 %tmp196200, 64		; <i32> [#uses=1]
	%tmp203 = and i32 %tmp199, -65		; <i32> [#uses=1]
	%tmp204 = or i32 %tmp203, %tmp201		; <i32> [#uses=1]
	store i32 %tmp204, ptr %tmp198
	%tmp206 = getelementptr %struct.attr_desc, ptr null, i32 0, i32 4		; <ptr> [#uses=1]
	%tmp207 = load i32, ptr %tmp206		; <i32> [#uses=1]
	%tmp211 = getelementptr %struct.attr_desc, ptr null, i32 0, i32 4		; <ptr> [#uses=2]
	%tmp212 = load i32, ptr %tmp211		; <i32> [#uses=1]
	%tmp209213 = shl i32 %tmp207, 1		; <i32> [#uses=1]
	%tmp214 = and i32 %tmp209213, 128		; <i32> [#uses=1]
	%tmp216 = and i32 %tmp212, -129		; <i32> [#uses=1]
	%tmp217 = or i32 %tmp216, %tmp214		; <i32> [#uses=1]
	store i32 %tmp217, ptr %tmp211
	ret void
}
