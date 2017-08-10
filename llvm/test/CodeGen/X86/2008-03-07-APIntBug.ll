; RUN: llc < %s -mtriple=i686-- -mcpu=i386 | not grep 255

	%struct.CONSTRAINT = type { i32, i32, i32, i32 }
	%struct.FIRST_UNION = type { %struct.anon }
	%struct.FOURTH_UNION = type { %struct.CONSTRAINT }
	%struct.LIST = type { %struct.rec*, %struct.rec* }
	%struct.SECOND_UNION = type { { i16, i8, i8 } }
	%struct.THIRD_UNION = type { { [2 x i32], [2 x i32] } }
	%struct.anon = type { i8, i8, i32 }
	%struct.head_type = type { [2 x %struct.LIST], %struct.FIRST_UNION, %struct.SECOND_UNION, %struct.THIRD_UNION, %struct.FOURTH_UNION, %struct.rec*, { %struct.rec* }, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, i32 }
	%struct.rec = type { %struct.head_type }
	%struct.symbol_type = type <{ [2 x %struct.LIST], %struct.FIRST_UNION, %struct.SECOND_UNION, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, i16, i16, i8, i8, i8, i8 }>
	%struct.word_type = type { [2 x %struct.LIST], %struct.FIRST_UNION, %struct.SECOND_UNION, %struct.THIRD_UNION, [4 x i8] }

define void @InsertSym_bb1163(%struct.rec** %s) {
newFuncRoot:
	br label %bb1163
bb1233.exitStub:		; preds = %bb1163
	ret void
bb1163:		; preds = %newFuncRoot
	%tmp1164 = load %struct.rec*, %struct.rec** %s, align 4		; <%struct.rec*> [#uses=1]
	%tmp1165 = getelementptr %struct.rec, %struct.rec* %tmp1164, i32 0, i32 0		; <%struct.head_type*> [#uses=1]
	%tmp11651166 = bitcast %struct.head_type* %tmp1165 to %struct.symbol_type*		; <%struct.symbol_type*> [#uses=1]
	%tmp1167 = getelementptr %struct.symbol_type, %struct.symbol_type* %tmp11651166, i32 0, i32 3		; <%struct.rec**> [#uses=1]
	%tmp1168 = load %struct.rec*, %struct.rec** %tmp1167, align 1		; <%struct.rec*> [#uses=2]
	%tmp1169 = load %struct.rec*, %struct.rec** %s, align 4		; <%struct.rec*> [#uses=1]
	%tmp1170 = getelementptr %struct.rec, %struct.rec* %tmp1169, i32 0, i32 0		; <%struct.head_type*> [#uses=1]
	%tmp11701171 = bitcast %struct.head_type* %tmp1170 to %struct.symbol_type*		; <%struct.symbol_type*> [#uses=1]
	%tmp1172 = getelementptr %struct.symbol_type, %struct.symbol_type* %tmp11701171, i32 0, i32 3		; <%struct.rec**> [#uses=1]
	%tmp1173 = load %struct.rec*, %struct.rec** %tmp1172, align 1		; <%struct.rec*> [#uses=2]
	%tmp1174 = getelementptr %struct.rec, %struct.rec* %tmp1173, i32 0, i32 0		; <%struct.head_type*> [#uses=1]
	%tmp11741175 = bitcast %struct.head_type* %tmp1174 to %struct.word_type*		; <%struct.word_type*> [#uses=1]
	%tmp1176 = getelementptr %struct.word_type, %struct.word_type* %tmp11741175, i32 0, i32 2		; <%struct.SECOND_UNION*> [#uses=1]
	%tmp1177 = getelementptr %struct.SECOND_UNION, %struct.SECOND_UNION* %tmp1176, i32 0, i32 0		; <{ i16, i8, i8 }*> [#uses=1]
	%tmp11771178 = bitcast { i16, i8, i8 }* %tmp1177 to <{ i8, i8, i8, i8 }>*		; <<{ i8, i8, i8, i8 }>*> [#uses=1]
	%tmp1179 = getelementptr <{ i8, i8, i8, i8 }>, <{ i8, i8, i8, i8 }>* %tmp11771178, i32 0, i32 2		; <i8*> [#uses=2]
	%mask1180 = and i8 1, 1		; <i8> [#uses=2]
	%tmp1181 = load i8, i8* %tmp1179, align 1		; <i8> [#uses=1]
	%tmp1182 = shl i8 %mask1180, 7		; <i8> [#uses=1]
	%tmp1183 = and i8 %tmp1181, 127		; <i8> [#uses=1]
	%tmp1184 = or i8 %tmp1183, %tmp1182		; <i8> [#uses=1]
	store i8 %tmp1184, i8* %tmp1179, align 1
	%mask1185 = and i8 %mask1180, 1		; <i8> [#uses=0]
	%tmp1186 = getelementptr %struct.rec, %struct.rec* %tmp1173, i32 0, i32 0		; <%struct.head_type*> [#uses=1]
	%tmp11861187 = bitcast %struct.head_type* %tmp1186 to %struct.word_type*		; <%struct.word_type*> [#uses=1]
	%tmp1188 = getelementptr %struct.word_type, %struct.word_type* %tmp11861187, i32 0, i32 2		; <%struct.SECOND_UNION*> [#uses=1]
	%tmp1189 = getelementptr %struct.SECOND_UNION, %struct.SECOND_UNION* %tmp1188, i32 0, i32 0		; <{ i16, i8, i8 }*> [#uses=1]
	%tmp11891190 = bitcast { i16, i8, i8 }* %tmp1189 to <{ i8, i8, i8, i8 }>*		; <<{ i8, i8, i8, i8 }>*> [#uses=1]
	%tmp1191 = getelementptr <{ i8, i8, i8, i8 }>, <{ i8, i8, i8, i8 }>* %tmp11891190, i32 0, i32 2		; <i8*> [#uses=1]
	%tmp1192 = load i8, i8* %tmp1191, align 1		; <i8> [#uses=1]
	%tmp1193 = lshr i8 %tmp1192, 7		; <i8> [#uses=1]
	%mask1194 = and i8 %tmp1193, 1		; <i8> [#uses=2]
	%mask1195 = and i8 %mask1194, 1		; <i8> [#uses=0]
	%tmp1196 = getelementptr %struct.rec, %struct.rec* %tmp1168, i32 0, i32 0		; <%struct.head_type*> [#uses=1]
	%tmp11961197 = bitcast %struct.head_type* %tmp1196 to %struct.word_type*		; <%struct.word_type*> [#uses=1]
	%tmp1198 = getelementptr %struct.word_type, %struct.word_type* %tmp11961197, i32 0, i32 2		; <%struct.SECOND_UNION*> [#uses=1]
	%tmp1199 = getelementptr %struct.SECOND_UNION, %struct.SECOND_UNION* %tmp1198, i32 0, i32 0		; <{ i16, i8, i8 }*> [#uses=1]
	%tmp11991200 = bitcast { i16, i8, i8 }* %tmp1199 to <{ i8, i8, i8, i8 }>*		; <<{ i8, i8, i8, i8 }>*> [#uses=1]
	%tmp1201 = getelementptr <{ i8, i8, i8, i8 }>, <{ i8, i8, i8, i8 }>* %tmp11991200, i32 0, i32 1		; <i8*> [#uses=2]
	%mask1202 = and i8 %mask1194, 1		; <i8> [#uses=2]
	%tmp1203 = load i8, i8* %tmp1201, align 1		; <i8> [#uses=1]
	%tmp1204 = shl i8 %mask1202, 1		; <i8> [#uses=1]
	%tmp1205 = and i8 %tmp1204, 2		; <i8> [#uses=1]
	%tmp1206 = and i8 %tmp1203, -3		; <i8> [#uses=1]
	%tmp1207 = or i8 %tmp1206, %tmp1205		; <i8> [#uses=1]
	store i8 %tmp1207, i8* %tmp1201, align 1
	%mask1208 = and i8 %mask1202, 1		; <i8> [#uses=0]
	%tmp1209 = getelementptr %struct.rec, %struct.rec* %tmp1168, i32 0, i32 0		; <%struct.head_type*> [#uses=1]
	%tmp12091210 = bitcast %struct.head_type* %tmp1209 to %struct.word_type*		; <%struct.word_type*> [#uses=1]
	%tmp1211 = getelementptr %struct.word_type, %struct.word_type* %tmp12091210, i32 0, i32 2		; <%struct.SECOND_UNION*> [#uses=1]
	%tmp1212 = getelementptr %struct.SECOND_UNION, %struct.SECOND_UNION* %tmp1211, i32 0, i32 0		; <{ i16, i8, i8 }*> [#uses=1]
	%tmp12121213 = bitcast { i16, i8, i8 }* %tmp1212 to <{ i8, i8, i8, i8 }>*		; <<{ i8, i8, i8, i8 }>*> [#uses=1]
	%tmp1214 = getelementptr <{ i8, i8, i8, i8 }>, <{ i8, i8, i8, i8 }>* %tmp12121213, i32 0, i32 1		; <i8*> [#uses=1]
	%tmp1215 = load i8, i8* %tmp1214, align 1		; <i8> [#uses=1]
	%tmp1216 = shl i8 %tmp1215, 6		; <i8> [#uses=1]
	%tmp1217 = lshr i8 %tmp1216, 7		; <i8> [#uses=1]
	%mask1218 = and i8 %tmp1217, 1		; <i8> [#uses=2]
	%mask1219 = and i8 %mask1218, 1		; <i8> [#uses=0]
	%tmp1220 = load %struct.rec*, %struct.rec** %s, align 4		; <%struct.rec*> [#uses=1]
	%tmp1221 = getelementptr %struct.rec, %struct.rec* %tmp1220, i32 0, i32 0		; <%struct.head_type*> [#uses=1]
	%tmp12211222 = bitcast %struct.head_type* %tmp1221 to %struct.word_type*		; <%struct.word_type*> [#uses=1]
	%tmp1223 = getelementptr %struct.word_type, %struct.word_type* %tmp12211222, i32 0, i32 2		; <%struct.SECOND_UNION*> [#uses=1]
	%tmp1224 = getelementptr %struct.SECOND_UNION, %struct.SECOND_UNION* %tmp1223, i32 0, i32 0		; <{ i16, i8, i8 }*> [#uses=1]
	%tmp12241225 = bitcast { i16, i8, i8 }* %tmp1224 to <{ i8, i8, i8, i8 }>*		; <<{ i8, i8, i8, i8 }>*> [#uses=1]
	%tmp1226 = getelementptr <{ i8, i8, i8, i8 }>, <{ i8, i8, i8, i8 }>* %tmp12241225, i32 0, i32 1		; <i8*> [#uses=2]
	%mask1227 = and i8 %mask1218, 1		; <i8> [#uses=2]
	%tmp1228 = load i8, i8* %tmp1226, align 1		; <i8> [#uses=1]
	%tmp1229 = and i8 %mask1227, 1		; <i8> [#uses=1]
	%tmp1230 = and i8 %tmp1228, -2		; <i8> [#uses=1]
	%tmp1231 = or i8 %tmp1230, %tmp1229		; <i8> [#uses=1]
	store i8 %tmp1231, i8* %tmp1226, align 1
	%mask1232 = and i8 %mask1227, 1		; <i8> [#uses=0]
	br label %bb1233.exitStub
}
