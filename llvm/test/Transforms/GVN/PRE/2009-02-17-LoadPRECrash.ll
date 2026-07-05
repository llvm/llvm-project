; RUN: opt < %s -passes=gvn -enable-load-pre -disable-output

	%struct.VEC_rtx_base = type { i32, i32, [1 x ptr] }
	%struct.VEC_rtx_gc = type { %struct.VEC_rtx_base }
	%struct.block_symbol = type { [3 x %struct.cgraph_rtl_info], ptr, i64 }
	%struct.cgraph_rtl_info = type { i32 }
	%struct.object_block = type { ptr, i32, i64, ptr, ptr }
	%struct.rtvec_def = type { i32, [1 x ptr] }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.section = type { %struct.unnamed_section }
	%struct.u = type { %struct.block_symbol }
	%struct.unnamed_section = type { %struct.cgraph_rtl_info, ptr, ptr, ptr }

declare ptr @gen_rtvec(i32, ...)

declare ptr @plus_constant(ptr, i64)

declare ptr @gen_rtx_fmt_Ei(i32, i32, ptr, i32)

declare i32 @local_symbolic_operand(ptr, i32)

define ptr @legitimize_pic_address(ptr %orig, ptr %reg) nounwind {
entry:
	%addr = alloca ptr		; <ptr> [#uses=5]
	%iftmp.1532 = alloca ptr		; <ptr> [#uses=3]
	store ptr %orig, ptr null
	%0 = load ptr, ptr null, align 4		; <ptr> [#uses=0]
	br i1 false, label %bb96, label %bb59

bb59:		; preds = %entry
	%1 = load ptr, ptr %addr, align 4		; <ptr> [#uses=1]
	%2 = call i32 @local_symbolic_operand(ptr %1, i32 0) nounwind		; <i32> [#uses=0]
	br i1 false, label %bb96, label %bb63

bb63:		; preds = %bb59
	br i1 false, label %bb64, label %bb74

bb64:		; preds = %bb63
	br i1 false, label %bb72, label %bb65

bb65:		; preds = %bb64
	br label %bb72

bb72:		; preds = %bb65, %bb64
	br label %bb74

bb74:		; preds = %bb72, %bb63
	br i1 false, label %bb75, label %bb76

bb75:		; preds = %bb74
	br label %bb76

bb76:		; preds = %bb75, %bb74
	br i1 false, label %bb77, label %bb84

bb77:		; preds = %bb76
	%3 = getelementptr [1 x %struct.cgraph_rtl_info], ptr null, i32 0, i32 0		; <ptr> [#uses=0]
	unreachable

bb84:		; preds = %bb76
	br i1 false, label %bb85, label %bb86

bb85:		; preds = %bb84
	br label %bb87

bb86:		; preds = %bb84
	br label %bb87

bb87:		; preds = %bb86, %bb85
	%4 = call ptr @gen_rtx_fmt_Ei(i32 16, i32 0, ptr null, i32 1) nounwind		; <ptr> [#uses=0]
	br i1 false, label %bb89, label %bb90

bb89:		; preds = %bb87
	br label %bb91

bb90:		; preds = %bb87
	br label %bb91

bb91:		; preds = %bb90, %bb89
	br i1 false, label %bb92, label %bb93

bb92:		; preds = %bb91
	br label %bb94

bb93:		; preds = %bb91
	br label %bb94

bb94:		; preds = %bb93, %bb92
	unreachable

bb96:		; preds = %bb59, %entry
	%5 = load ptr, ptr %addr, align 4		; <ptr> [#uses=1]
	%6 = getelementptr %struct.rtx_def, ptr %5, i32 0, i32 0		; <ptr> [#uses=1]
	%7 = load i16, ptr %6, align 2		; <i16> [#uses=0]
	br i1 false, label %bb147, label %bb97

bb97:		; preds = %bb96
	%8 = load ptr, ptr %addr, align 4		; <ptr> [#uses=0]
	br i1 false, label %bb147, label %bb99

bb99:		; preds = %bb97
	unreachable

bb147:		; preds = %bb97, %bb96
	%9 = load ptr, ptr %addr, align 4		; <ptr> [#uses=1]
	%10 = getelementptr %struct.rtx_def, ptr %9, i32 0, i32 0		; <ptr> [#uses=1]
	%11 = load i16, ptr %10, align 2		; <i16> [#uses=0]
	br i1 false, label %bb164, label %bb148

bb148:		; preds = %bb147
	br i1 false, label %bb164, label %bb149

bb149:		; preds = %bb148
	br i1 false, label %bb150, label %bb152

bb150:		; preds = %bb149
	unreachable

bb152:		; preds = %bb149
	br label %bb164

bb164:		; preds = %bb152, %bb148, %bb147
	%12 = getelementptr [1 x %struct.cgraph_rtl_info], ptr null, i32 0, i32 1		; <ptr> [#uses=0]
	br i1 false, label %bb165, label %bb166

bb165:		; preds = %bb164
	br label %bb167

bb166:		; preds = %bb164
	br label %bb167

bb167:		; preds = %bb166, %bb165
	br i1 false, label %bb211, label %bb168

bb168:		; preds = %bb167
	br i1 false, label %bb211, label %bb170

bb170:		; preds = %bb168
	br i1 false, label %bb172, label %bb181

bb172:		; preds = %bb170
	br i1 false, label %bb179, label %bb174

bb174:		; preds = %bb172
	br i1 false, label %bb177, label %bb175

bb175:		; preds = %bb174
	br i1 false, label %bb177, label %bb176

bb176:		; preds = %bb175
	br label %bb178

bb177:		; preds = %bb175, %bb174
	br label %bb178

bb178:		; preds = %bb177, %bb176
	br label %bb180

bb179:		; preds = %bb172
	br label %bb180

bb180:		; preds = %bb179, %bb178
	br label %bb181

bb181:		; preds = %bb180, %bb170
	%13 = call ptr (i32, ...) @gen_rtvec(i32 1, ptr null) nounwind		; <ptr> [#uses=0]
	unreachable

bb211:		; preds = %bb168, %bb167
	%14 = load ptr, ptr %addr, align 4		; <ptr> [#uses=0]
	%15 = getelementptr [1 x %struct.cgraph_rtl_info], ptr null, i32 0, i32 0		; <ptr> [#uses=0]
	store ptr null, ptr null, align 4
	br i1 false, label %bb212, label %bb213

bb212:		; preds = %bb211
	store ptr null, ptr %iftmp.1532, align 4
	br label %bb214

bb213:		; preds = %bb211
	store ptr null, ptr %iftmp.1532, align 4
	br label %bb214

bb214:		; preds = %bb213, %bb212
	%16 = getelementptr [1 x %struct.cgraph_rtl_info], ptr null, i32 0, i32 1		; <ptr> [#uses=0]
	%17 = load ptr, ptr %iftmp.1532, align 4		; <ptr> [#uses=0]
	%18 = getelementptr %struct.rtx_def, ptr null, i32 0, i32 3		; <ptr> [#uses=1]
	%19 = getelementptr %struct.u, ptr %18, i32 0, i32 0		; <ptr> [#uses=1]
	%20 = getelementptr [1 x i64], ptr %19, i32 0, i32 0		; <ptr> [#uses=0]
	%21 = call ptr @plus_constant(ptr null, i64 0) nounwind		; <ptr> [#uses=0]
	unreachable
}
