; RUN: llvm-as < %s | opt -instcombine -disable-output
; END.
	%struct.gs_matrix = type { float, i32, float, i32, float, i32, float, i32, float, i32, float, i32 }
	%struct.gx_bitmap = type { i8*, i32, i32, i32 }
	%struct.gx_device = type { i32, %struct.gx_device_procs*, i8*, i32, i32, float, float, i32, i16, i32, i32 }
	%struct.gx_device_memory = type { i32, %struct.gx_device_procs*, i8*, i32, i32, float, float, i32, i16, i32, i32, %struct.gs_matrix, i32, i8*, i8**, i32 (%struct.gx_device_memory*, i32, i32, i32, i32, i32)*, i32, i32, i8* }
	%struct.gx_device_procs = type { i32 (%struct.gx_device*)*, void (%struct.gx_device*, %struct.gs_matrix*)*, i32 (%struct.gx_device*)*, i32 (%struct.gx_device*)*, i32 (%struct.gx_device*)*, i32 (%struct.gx_device*, i16, i16, i16)*, i32 (%struct.gx_device*, i32, i16*)*, i32 (%struct.gx_device*, i32, i32, i32, i32, i32)*, i32 (%struct.gx_device*, %struct.gx_bitmap*, i32, i32, i32, i32, i32, i32)*, i32 (%struct.gx_device*, i8*, i32, i32, i32, i32, i32, i32, i32, i32)*, i32 (%struct.gx_device*, i8*, i32, i32, i32, i32, i32, i32)*, i32 (%struct.gx_device*, i32, i32, i32, i32, i32)*, i32 (%struct.gx_device*, i32, i32, i32, i32, i32, i32, i32)*, i32 (%struct.gx_device*, %struct.gx_bitmap*, i32, i32, i32, i32, i32, i32, i32, i32)* }

define i32 @mem_mono_copy_mono(%struct.gx_device* %dev, i8* %base, i32 %sourcex, i32 %raster, i32 %x, i32 %y, i32 %w, i32 %h, i32 %zero, i32 %one) {
entry:
	%tmp = icmp eq i32 %one, %zero		; <i1> [#uses=1]
	br i1 %tmp, label %cond_true, label %cond_next

cond_true:		; preds = %entry
	%tmp6 = tail call i32 @mem_mono_fill_rectangle( %struct.gx_device* %dev, i32 %x, i32 %y, i32 %w, i32 %h, i32 %zero )		; <i32> [#uses=1]
	ret i32 %tmp6

cond_next:		; preds = %entry
	%tmp8 = bitcast %struct.gx_device* %dev to %struct.gx_device_memory*		; <%struct.gx_device_memory*> [#uses=6]
	%tmp.upgrd.1 = getelementptr %struct.gx_device_memory* %tmp8, i32 0, i32 15		; <i32 (%struct.gx_device_memory*, i32, i32, i32, i32, i32)**> [#uses=1]
	%tmp.upgrd.2 = load i32 (%struct.gx_device_memory*, i32, i32, i32, i32, i32)** %tmp.upgrd.1		; <i32 (%struct.gx_device_memory*, i32, i32, i32, i32, i32)*> [#uses=2]
	%tmp9 = icmp eq i32 (%struct.gx_device_memory*, i32, i32, i32, i32, i32)* %tmp.upgrd.2, @mem_no_fault_proc		; <i1> [#uses=1]
	br i1 %tmp9, label %cond_next46, label %cond_true10

cond_true10:		; preds = %cond_next
	%tmp16 = add i32 %x, 7		; <i32> [#uses=1]
	%tmp17 = add i32 %tmp16, %w		; <i32> [#uses=1]
	%tmp18 = ashr i32 %tmp17, 3		; <i32> [#uses=1]
	%tmp20 = ashr i32 %x, 3		; <i32> [#uses=2]
	%tmp21 = sub i32 %tmp18, %tmp20		; <i32> [#uses=1]
	%tmp27 = tail call i32 %tmp.upgrd.2( %struct.gx_device_memory* %tmp8, i32 %tmp20, i32 %y, i32 %tmp21, i32 %h, i32 1 )		; <i32> [#uses=2]
	%tmp29 = icmp slt i32 %tmp27, 0		; <i1> [#uses=1]
	br i1 %tmp29, label %cond_true30, label %cond_next46

cond_true30:		; preds = %cond_true10
	%tmp41 = tail call i32 @mem_copy_mono_recover( %struct.gx_device* %dev, i8* %base, i32 %sourcex, i32 %raster, i32 %x, i32 %y, i32 %w, i32 %h, i32 %zero, i32 %one, i32 %tmp27 )		; <i32> [#uses=1]
	ret i32 %tmp41

cond_next46:		; preds = %cond_true10, %cond_next
	%tmp48 = icmp sgt i32 %w, 0		; <i1> [#uses=1]
	%tmp53 = icmp sgt i32 %h, 0		; <i1> [#uses=1]
	%bothcond = and i1 %tmp53, %tmp48		; <i1> [#uses=1]
	br i1 %bothcond, label %bb58, label %return

bb58:		; preds = %cond_next46
	%tmp60 = icmp slt i32 %x, 0		; <i1> [#uses=1]
	br i1 %tmp60, label %return, label %cond_next63

cond_next63:		; preds = %bb58
	%tmp65 = getelementptr %struct.gx_device_memory* %tmp8, i32 0, i32 3		; <i32*> [#uses=1]
	%tmp66 = load i32* %tmp65		; <i32> [#uses=1]
	%tmp68 = sub i32 %tmp66, %w		; <i32> [#uses=1]
	%tmp70 = icmp slt i32 %tmp68, %x		; <i1> [#uses=1]
	%tmp75 = icmp slt i32 %y, 0		; <i1> [#uses=1]
	%bothcond1 = or i1 %tmp70, %tmp75		; <i1> [#uses=1]
	br i1 %bothcond1, label %return, label %cond_next78

cond_next78:		; preds = %cond_next63
	%tmp80 = getelementptr %struct.gx_device_memory* %tmp8, i32 0, i32 4		; <i32*> [#uses=1]
	%tmp81 = load i32* %tmp80		; <i32> [#uses=1]
	%tmp83 = sub i32 %tmp81, %h		; <i32> [#uses=1]
	%tmp85 = icmp slt i32 %tmp83, %y		; <i1> [#uses=1]
	br i1 %tmp85, label %return, label %bb91

bb91:		; preds = %cond_next78
	%tmp93 = ashr i32 %x, 3		; <i32> [#uses=4]
	%tmp.upgrd.3 = getelementptr %struct.gx_device_memory* %tmp8, i32 0, i32 14		; <i8***> [#uses=1]
	%tmp.upgrd.4 = load i8*** %tmp.upgrd.3		; <i8**> [#uses=1]
	%tmp96 = getelementptr i8** %tmp.upgrd.4, i32 %y		; <i8**> [#uses=4]
	%tmp98 = load i8** %tmp96		; <i8*> [#uses=1]
	%tmp100 = getelementptr i8* %tmp98, i32 %tmp93		; <i8*> [#uses=3]
	%tmp102 = ashr i32 %sourcex, 3		; <i32> [#uses=3]
	%tmp106 = and i32 %sourcex, 7		; <i32> [#uses=1]
	%tmp107 = sub i32 8, %tmp106		; <i32> [#uses=4]
	%tmp109 = and i32 %x, 7		; <i32> [#uses=3]
	%tmp110 = sub i32 8, %tmp109		; <i32> [#uses=8]
	%tmp112 = sub i32 8, %tmp110		; <i32> [#uses=1]
	%tmp112.upgrd.5 = trunc i32 %tmp112 to i8		; <i8> [#uses=1]
	%shift.upgrd.6 = zext i8 %tmp112.upgrd.5 to i32		; <i32> [#uses=1]
	%tmp113464 = lshr i32 255, %shift.upgrd.6		; <i32> [#uses=4]
	%tmp116 = icmp sgt i32 %tmp110, %w		; <i1> [#uses=1]
	%tmp132 = getelementptr %struct.gx_device_memory* %tmp8, i32 0, i32 16		; <i32*> [#uses=2]
	br i1 %tmp116, label %cond_true117, label %cond_false123

cond_true117:		; preds = %bb91
	%tmp119 = trunc i32 %w to i8		; <i8> [#uses=1]
	%shift.upgrd.7 = zext i8 %tmp119 to i32		; <i32> [#uses=1]
	%tmp120 = lshr i32 %tmp113464, %shift.upgrd.7		; <i32> [#uses=1]
	%tmp122 = sub i32 %tmp113464, %tmp120		; <i32> [#uses=2]
	%tmp13315 = load i32* %tmp132		; <i32> [#uses=1]
	%tmp13416 = icmp eq i32 %tmp13315, 0		; <i1> [#uses=1]
	br i1 %tmp13416, label %cond_next151, label %cond_true135

cond_false123:		; preds = %bb91
	%tmp126 = sub i32 %w, %tmp110		; <i32> [#uses=1]
	%tmp126.upgrd.8 = trunc i32 %tmp126 to i8		; <i8> [#uses=1]
	%tmp127 = and i8 %tmp126.upgrd.8, 7		; <i8> [#uses=1]
	%shift.upgrd.9 = zext i8 %tmp127 to i32		; <i32> [#uses=1]
	%tmp128 = lshr i32 255, %shift.upgrd.9		; <i32> [#uses=1]
	%tmp1295 = sub i32 255, %tmp128		; <i32> [#uses=2]
	%tmp133 = load i32* %tmp132		; <i32> [#uses=1]
	%tmp134 = icmp eq i32 %tmp133, 0		; <i1> [#uses=1]
	br i1 %tmp134, label %cond_next151, label %cond_true135

cond_true135:		; preds = %cond_false123, %cond_true117
	%rmask.0.0 = phi i32 [ undef, %cond_true117 ], [ %tmp1295, %cond_false123 ]		; <i32> [#uses=2]
	%mask.1.0 = phi i32 [ %tmp122, %cond_true117 ], [ %tmp113464, %cond_false123 ]		; <i32> [#uses=2]
	%not.tmp137 = icmp ne i32 %zero, -1		; <i1> [#uses=1]
	%tmp140 = zext i1 %not.tmp137 to i32		; <i32> [#uses=1]
	%zero_addr.0 = xor i32 %tmp140, %zero		; <i32> [#uses=2]
	%tmp144 = icmp eq i32 %one, -1		; <i1> [#uses=1]
	br i1 %tmp144, label %cond_next151, label %cond_true145

cond_true145:		; preds = %cond_true135
	%tmp147 = xor i32 %one, 1		; <i32> [#uses=1]
	br label %cond_next151

cond_next151:		; preds = %cond_true145, %cond_true135, %cond_false123, %cond_true117
	%rmask.0.1 = phi i32 [ %rmask.0.0, %cond_true145 ], [ undef, %cond_true117 ], [ %tmp1295, %cond_false123 ], [ %rmask.0.0, %cond_true135 ]		; <i32> [#uses=4]
	%mask.1.1 = phi i32 [ %mask.1.0, %cond_true145 ], [ %tmp122, %cond_true117 ], [ %tmp113464, %cond_false123 ], [ %mask.1.0, %cond_true135 ]		; <i32> [#uses=4]
	%one_addr.0 = phi i32 [ %tmp147, %cond_true145 ], [ %one, %cond_true117 ], [ %one, %cond_false123 ], [ %one, %cond_true135 ]		; <i32> [#uses=2]
	%zero_addr.1 = phi i32 [ %zero_addr.0, %cond_true145 ], [ %zero, %cond_true117 ], [ %zero, %cond_false123 ], [ %zero_addr.0, %cond_true135 ]		; <i32> [#uses=2]
	%tmp153 = icmp eq i32 %zero_addr.1, 1		; <i1> [#uses=2]
	%tmp158 = icmp eq i32 %one_addr.0, 0		; <i1> [#uses=2]
	%bothcond2 = or i1 %tmp153, %tmp158		; <i1> [#uses=1]
	%iftmp.35.0 = select i1 %bothcond2, i32 -1, i32 0		; <i32> [#uses=9]
	%tmp167 = icmp eq i32 %zero_addr.1, 0		; <i1> [#uses=1]
	%bothcond3 = or i1 %tmp167, %tmp158		; <i1> [#uses=1]
	%iftmp.36.0 = select i1 %bothcond3, i32 0, i32 -1		; <i32> [#uses=4]
	%tmp186 = icmp eq i32 %one_addr.0, 1		; <i1> [#uses=1]
	%bothcond4 = or i1 %tmp153, %tmp186		; <i1> [#uses=1]
	%iftmp.37.0 = select i1 %bothcond4, i32 -1, i32 0		; <i32> [#uses=6]
	%tmp196 = icmp eq i32 %tmp107, %tmp110		; <i1> [#uses=1]
	br i1 %tmp196, label %cond_true197, label %cond_false299

cond_true197:		; preds = %cond_next151
	%tmp29222 = add i32 %h, -1		; <i32> [#uses=3]
	%tmp29424 = icmp slt i32 %tmp29222, 0		; <i1> [#uses=1]
	br i1 %tmp29424, label %return, label %cond_true295.preheader

cond_true249.preheader:		; preds = %cond_true295
	br label %cond_true249

cond_true249:		; preds = %cond_true249, %cond_true249.preheader
	%indvar = phi i32 [ 0, %cond_true249.preheader ], [ %indvar.next, %cond_true249 ]		; <i32> [#uses=3]
	%optr.3.2 = phi i8* [ %tmp232.upgrd.12, %cond_true249 ], [ %dest.1.0, %cond_true249.preheader ]		; <i8*> [#uses=1]
	%bptr.3.2 = phi i8* [ %tmp226.upgrd.10, %cond_true249 ], [ %line.1.0, %cond_true249.preheader ]		; <i8*> [#uses=1]
	%tmp. = add i32 %tmp109, %w		; <i32> [#uses=1]
	%tmp.58 = mul i32 %indvar, -8		; <i32> [#uses=1]
	%tmp.57 = add i32 %tmp., -16		; <i32> [#uses=1]
	%tmp246.2 = add i32 %tmp.58, %tmp.57		; <i32> [#uses=1]
	%tmp225 = ptrtoint i8* %bptr.3.2 to i32		; <i32> [#uses=1]
	%tmp226 = add i32 %tmp225, 1		; <i32> [#uses=1]
	%tmp226.upgrd.10 = inttoptr i32 %tmp226 to i8*		; <i8*> [#uses=3]
	%tmp228 = load i8* %tmp226.upgrd.10		; <i8> [#uses=1]
	%tmp228.upgrd.11 = zext i8 %tmp228 to i32		; <i32> [#uses=1]
	%tmp230 = xor i32 %tmp228.upgrd.11, %iftmp.35.0		; <i32> [#uses=2]
	%tmp231 = ptrtoint i8* %optr.3.2 to i32		; <i32> [#uses=1]
	%tmp232 = add i32 %tmp231, 1		; <i32> [#uses=1]
	%tmp232.upgrd.12 = inttoptr i32 %tmp232 to i8*		; <i8*> [#uses=4]
	%tmp235 = or i32 %tmp230, %iftmp.36.0		; <i32> [#uses=1]
	%tmp235.upgrd.13 = trunc i32 %tmp235 to i8		; <i8> [#uses=1]
	%tmp237 = load i8* %tmp232.upgrd.12		; <i8> [#uses=1]
	%tmp238 = and i8 %tmp235.upgrd.13, %tmp237		; <i8> [#uses=1]
	%tmp241 = and i32 %tmp230, %iftmp.37.0		; <i32> [#uses=1]
	%tmp241.upgrd.14 = trunc i32 %tmp241 to i8		; <i8> [#uses=1]
	%tmp242 = or i8 %tmp238, %tmp241.upgrd.14		; <i8> [#uses=1]
	store i8 %tmp242, i8* %tmp232.upgrd.12
	%tmp24629 = add i32 %tmp246.2, -8		; <i32> [#uses=2]
	%tmp24831 = icmp slt i32 %tmp24629, 0		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp24831, label %bb252.loopexit, label %cond_true249

bb252.loopexit:		; preds = %cond_true249
	br label %bb252

bb252:		; preds = %cond_true295, %bb252.loopexit
	%optr.3.3 = phi i8* [ %dest.1.0, %cond_true295 ], [ %tmp232.upgrd.12, %bb252.loopexit ]		; <i8*> [#uses=1]
	%bptr.3.3 = phi i8* [ %line.1.0, %cond_true295 ], [ %tmp226.upgrd.10, %bb252.loopexit ]		; <i8*> [#uses=1]
	%tmp246.3 = phi i32 [ %tmp246, %cond_true295 ], [ %tmp24629, %bb252.loopexit ]		; <i32> [#uses=1]
	%tmp254 = icmp sgt i32 %tmp246.3, -8		; <i1> [#uses=1]
	br i1 %tmp254, label %cond_true255, label %cond_next280

cond_true255:		; preds = %bb252
	%tmp256 = ptrtoint i8* %bptr.3.3 to i32		; <i32> [#uses=1]
	%tmp257 = add i32 %tmp256, 1		; <i32> [#uses=1]
	%tmp257.upgrd.15 = inttoptr i32 %tmp257 to i8*		; <i8*> [#uses=1]
	%tmp259 = load i8* %tmp257.upgrd.15		; <i8> [#uses=1]
	%tmp259.upgrd.16 = zext i8 %tmp259 to i32		; <i32> [#uses=1]
	%tmp261 = xor i32 %tmp259.upgrd.16, %iftmp.35.0		; <i32> [#uses=2]
	%tmp262 = ptrtoint i8* %optr.3.3 to i32		; <i32> [#uses=1]
	%tmp263 = add i32 %tmp262, 1		; <i32> [#uses=1]
	%tmp263.upgrd.17 = inttoptr i32 %tmp263 to i8*		; <i8*> [#uses=2]
	%tmp265 = trunc i32 %tmp261 to i8		; <i8> [#uses=1]
	%tmp268 = or i8 %tmp266, %tmp265		; <i8> [#uses=1]
	%tmp270 = load i8* %tmp263.upgrd.17		; <i8> [#uses=1]
	%tmp271 = and i8 %tmp268, %tmp270		; <i8> [#uses=1]
	%tmp276 = and i32 %tmp274, %tmp261		; <i32> [#uses=1]
	%tmp276.upgrd.18 = trunc i32 %tmp276 to i8		; <i8> [#uses=1]
	%tmp277 = or i8 %tmp271, %tmp276.upgrd.18		; <i8> [#uses=1]
	store i8 %tmp277, i8* %tmp263.upgrd.17
	br label %cond_next280

cond_next280:		; preds = %cond_true255, %bb252
	%tmp281 = ptrtoint i8** %dest_line.1.0 to i32		; <i32> [#uses=1]
	%tmp282 = add i32 %tmp281, 4		; <i32> [#uses=1]
	%tmp282.upgrd.19 = inttoptr i32 %tmp282 to i8**		; <i8**> [#uses=2]
	%tmp284 = load i8** %tmp282.upgrd.19		; <i8*> [#uses=1]
	%tmp286 = getelementptr i8* %tmp284, i32 %tmp93		; <i8*> [#uses=1]
	%tmp292 = add i32 %tmp292.0, -1		; <i32> [#uses=1]
	%tmp294 = icmp slt i32 %tmp292, 0		; <i1> [#uses=1]
	%indvar.next61 = add i32 %indvar60, 1		; <i32> [#uses=1]
	br i1 %tmp294, label %return.loopexit, label %cond_true295

cond_true295.preheader:		; preds = %cond_true197
	%tmp200 = sub i32 %w, %tmp110		; <i32> [#uses=1]
	%tmp209 = trunc i32 %mask.1.1 to i8		; <i8> [#uses=1]
	%tmp209not = xor i8 %tmp209, -1		; <i8> [#uses=1]
	%tmp212 = trunc i32 %iftmp.36.0 to i8		; <i8> [#uses=2]
	%tmp211 = or i8 %tmp212, %tmp209not		; <i8> [#uses=2]
	%tmp219 = and i32 %iftmp.37.0, %mask.1.1		; <i32> [#uses=2]
	%tmp246 = add i32 %tmp200, -8		; <i32> [#uses=3]
	%tmp248 = icmp slt i32 %tmp246, 0		; <i1> [#uses=1]
	%tmp264 = trunc i32 %rmask.0.1 to i8		; <i8> [#uses=1]
	%tmp264not = xor i8 %tmp264, -1		; <i8> [#uses=1]
	%tmp266 = or i8 %tmp212, %tmp264not		; <i8> [#uses=2]
	%tmp274 = and i32 %iftmp.37.0, %rmask.0.1		; <i32> [#uses=2]
	br i1 %tmp248, label %cond_true295.preheader.split.us, label %cond_true295.preheader.split

cond_true295.preheader.split.us:		; preds = %cond_true295.preheader
	br label %cond_true295.us

cond_true295.us:		; preds = %cond_next280.us, %cond_true295.preheader.split.us
	%indvar86 = phi i32 [ 0, %cond_true295.preheader.split.us ], [ %indvar.next87, %cond_next280.us ]		; <i32> [#uses=3]
	%dest.1.0.us = phi i8* [ %tmp286.us, %cond_next280.us ], [ %tmp100, %cond_true295.preheader.split.us ]		; <i8*> [#uses=3]
	%dest_line.1.0.us = phi i8** [ %tmp282.us.upgrd.21, %cond_next280.us ], [ %tmp96, %cond_true295.preheader.split.us ]		; <i8**> [#uses=1]
	%tmp.89 = sub i32 0, %indvar86		; <i32> [#uses=2]
	%tmp292.0.us = add i32 %tmp.89, %tmp29222		; <i32> [#uses=1]
	%tmp.91 = mul i32 %indvar86, %raster		; <i32> [#uses=2]
	%tmp104.sum101 = add i32 %tmp102, %tmp.91		; <i32> [#uses=1]
	%line.1.0.us = getelementptr i8* %base, i32 %tmp104.sum101		; <i8*> [#uses=2]
	%tmp.us = load i8* %line.1.0.us		; <i8> [#uses=1]
	%tmp206.us = zext i8 %tmp.us to i32		; <i32> [#uses=1]
	%tmp208.us = xor i32 %tmp206.us, %iftmp.35.0		; <i32> [#uses=2]
	%tmp210.us = trunc i32 %tmp208.us to i8		; <i8> [#uses=1]
	%tmp213.us = or i8 %tmp211, %tmp210.us		; <i8> [#uses=1]
	%tmp215.us = load i8* %dest.1.0.us		; <i8> [#uses=1]
	%tmp216.us = and i8 %tmp213.us, %tmp215.us		; <i8> [#uses=1]
	%tmp221.us = and i32 %tmp219, %tmp208.us		; <i32> [#uses=1]
	%tmp221.us.upgrd.20 = trunc i32 %tmp221.us to i8		; <i8> [#uses=1]
	%tmp222.us = or i8 %tmp216.us, %tmp221.us.upgrd.20		; <i8> [#uses=1]
	store i8 %tmp222.us, i8* %dest.1.0.us
	br i1 true, label %bb252.us, label %cond_true249.preheader.us

cond_next280.us:		; preds = %bb252.us, %cond_true255.us
	%tmp281.us = ptrtoint i8** %dest_line.1.0.us to i32		; <i32> [#uses=1]
	%tmp282.us = add i32 %tmp281.us, 4		; <i32> [#uses=1]
	%tmp282.us.upgrd.21 = inttoptr i32 %tmp282.us to i8**		; <i8**> [#uses=2]
	%tmp284.us = load i8** %tmp282.us.upgrd.21		; <i8*> [#uses=1]
	%tmp286.us = getelementptr i8* %tmp284.us, i32 %tmp93		; <i8*> [#uses=1]
	%tmp292.us = add i32 %tmp292.0.us, -1		; <i32> [#uses=1]
	%tmp294.us = icmp slt i32 %tmp292.us, 0		; <i1> [#uses=1]
	%indvar.next87 = add i32 %indvar86, 1		; <i32> [#uses=1]
	br i1 %tmp294.us, label %return.loopexit.us, label %cond_true295.us

cond_true255.us:		; preds = %bb252.us
	%tmp256.us = ptrtoint i8* %bptr.3.3.us to i32		; <i32> [#uses=1]
	%tmp257.us = add i32 %tmp256.us, 1		; <i32> [#uses=1]
	%tmp257.us.upgrd.22 = inttoptr i32 %tmp257.us to i8*		; <i8*> [#uses=1]
	%tmp259.us = load i8* %tmp257.us.upgrd.22		; <i8> [#uses=1]
	%tmp259.us.upgrd.23 = zext i8 %tmp259.us to i32		; <i32> [#uses=1]
	%tmp261.us = xor i32 %tmp259.us.upgrd.23, %iftmp.35.0		; <i32> [#uses=2]
	%tmp262.us = ptrtoint i8* %optr.3.3.us to i32		; <i32> [#uses=1]
	%tmp263.us = add i32 %tmp262.us, 1		; <i32> [#uses=1]
	%tmp263.us.upgrd.24 = inttoptr i32 %tmp263.us to i8*		; <i8*> [#uses=2]
	%tmp265.us = trunc i32 %tmp261.us to i8		; <i8> [#uses=1]
	%tmp268.us = or i8 %tmp266, %tmp265.us		; <i8> [#uses=1]
	%tmp270.us = load i8* %tmp263.us.upgrd.24		; <i8> [#uses=1]
	%tmp271.us = and i8 %tmp268.us, %tmp270.us		; <i8> [#uses=1]
	%tmp276.us = and i32 %tmp274, %tmp261.us		; <i32> [#uses=1]
	%tmp276.us.upgrd.25 = trunc i32 %tmp276.us to i8		; <i8> [#uses=1]
	%tmp277.us = or i8 %tmp271.us, %tmp276.us.upgrd.25		; <i8> [#uses=1]
	store i8 %tmp277.us, i8* %tmp263.us.upgrd.24
	br label %cond_next280.us

bb252.us:		; preds = %bb252.loopexit.us, %cond_true295.us
	%optr.3.3.us = phi i8* [ %dest.1.0.us, %cond_true295.us ], [ undef, %bb252.loopexit.us ]		; <i8*> [#uses=1]
	%bptr.3.3.us = phi i8* [ %line.1.0.us, %cond_true295.us ], [ undef, %bb252.loopexit.us ]		; <i8*> [#uses=1]
	%tmp246.3.us = phi i32 [ %tmp246, %cond_true295.us ], [ undef, %bb252.loopexit.us ]		; <i32> [#uses=1]
	%tmp254.us = icmp sgt i32 %tmp246.3.us, -8		; <i1> [#uses=1]
	br i1 %tmp254.us, label %cond_true255.us, label %cond_next280.us

cond_true249.us:		; preds = %cond_true249.preheader.us, %cond_true249.us
	br i1 undef, label %bb252.loopexit.us, label %cond_true249.us

cond_true249.preheader.us:		; preds = %cond_true295.us
	br label %cond_true249.us

bb252.loopexit.us:		; preds = %cond_true249.us
	br label %bb252.us

return.loopexit.us:		; preds = %cond_next280.us
	br label %return.loopexit.split

cond_true295.preheader.split:		; preds = %cond_true295.preheader
	br label %cond_true295

cond_true295:		; preds = %cond_true295.preheader.split, %cond_next280
	%indvar60 = phi i32 [ 0, %cond_true295.preheader.split ], [ %indvar.next61, %cond_next280 ]		; <i32> [#uses=3]
	%dest.1.0 = phi i8* [ %tmp286, %cond_next280 ], [ %tmp100, %cond_true295.preheader.split ]		; <i8*> [#uses=4]
	%dest_line.1.0 = phi i8** [ %tmp282.upgrd.19, %cond_next280 ], [ %tmp96, %cond_true295.preheader.split ]		; <i8**> [#uses=1]
	%tmp.63 = sub i32 0, %indvar60		; <i32> [#uses=2]
	%tmp292.0 = add i32 %tmp.63, %tmp29222		; <i32> [#uses=1]
	%tmp.65 = mul i32 %indvar60, %raster		; <i32> [#uses=2]
	%tmp104.sum97 = add i32 %tmp102, %tmp.65		; <i32> [#uses=1]
	%line.1.0 = getelementptr i8* %base, i32 %tmp104.sum97		; <i8*> [#uses=3]
	%tmp.upgrd.26 = load i8* %line.1.0		; <i8> [#uses=1]
	%tmp206 = zext i8 %tmp.upgrd.26 to i32		; <i32> [#uses=1]
	%tmp208 = xor i32 %tmp206, %iftmp.35.0		; <i32> [#uses=2]
	%tmp210 = trunc i32 %tmp208 to i8		; <i8> [#uses=1]
	%tmp213 = or i8 %tmp211, %tmp210		; <i8> [#uses=1]
	%tmp215 = load i8* %dest.1.0		; <i8> [#uses=1]
	%tmp216 = and i8 %tmp213, %tmp215		; <i8> [#uses=1]
	%tmp221 = and i32 %tmp219, %tmp208		; <i32> [#uses=1]
	%tmp221.upgrd.27 = trunc i32 %tmp221 to i8		; <i8> [#uses=1]
	%tmp222 = or i8 %tmp216, %tmp221.upgrd.27		; <i8> [#uses=1]
	store i8 %tmp222, i8* %dest.1.0
	br i1 false, label %bb252, label %cond_true249.preheader

cond_false299:		; preds = %cond_next151
	%tmp302 = sub i32 %tmp107, %tmp110		; <i32> [#uses=1]
	%tmp303 = and i32 %tmp302, 7		; <i32> [#uses=3]
	%tmp305 = sub i32 8, %tmp303		; <i32> [#uses=1]
	%tmp45438 = add i32 %h, -1		; <i32> [#uses=2]
	%tmp45640 = icmp slt i32 %tmp45438, 0		; <i1> [#uses=1]
	br i1 %tmp45640, label %return, label %cond_true457.preheader

cond_true316:		; preds = %cond_true457
	%tmp318 = zext i8 %tmp318.upgrd.48 to i32		; <i32> [#uses=1]
	%shift.upgrd.28 = zext i8 %tmp319 to i32		; <i32> [#uses=1]
	%tmp320 = lshr i32 %tmp318, %shift.upgrd.28		; <i32> [#uses=1]
	br label %cond_next340

cond_false321:		; preds = %cond_true457
	%tmp3188 = zext i8 %tmp318.upgrd.48 to i32		; <i32> [#uses=1]
	%shift.upgrd.29 = zext i8 %tmp324 to i32		; <i32> [#uses=1]
	%tmp325 = shl i32 %tmp3188, %shift.upgrd.29		; <i32> [#uses=2]
	%tmp326 = ptrtoint i8* %line.3.0 to i32		; <i32> [#uses=1]
	%tmp327 = add i32 %tmp326, 1		; <i32> [#uses=1]
	%tmp327.upgrd.30 = inttoptr i32 %tmp327 to i8*		; <i8*> [#uses=3]
	br i1 %tmp330, label %cond_true331, label %cond_next340

cond_true331:		; preds = %cond_false321
	%tmp333 = load i8* %tmp327.upgrd.30		; <i8> [#uses=1]
	%tmp333.upgrd.31 = zext i8 %tmp333 to i32		; <i32> [#uses=1]
	%shift.upgrd.32 = zext i8 %tmp319 to i32		; <i32> [#uses=1]
	%tmp335 = lshr i32 %tmp333.upgrd.31, %shift.upgrd.32		; <i32> [#uses=1]
	%tmp337 = add i32 %tmp335, %tmp325		; <i32> [#uses=1]
	br label %cond_next340

cond_next340:		; preds = %cond_true331, %cond_false321, %cond_true316
	%bits.0 = phi i32 [ %tmp320, %cond_true316 ], [ %tmp337, %cond_true331 ], [ %tmp325, %cond_false321 ]		; <i32> [#uses=1]
	%bptr307.3 = phi i8* [ %line.3.0, %cond_true316 ], [ %tmp327.upgrd.30, %cond_true331 ], [ %tmp327.upgrd.30, %cond_false321 ]		; <i8*> [#uses=2]
	%tmp343 = xor i32 %bits.0, %iftmp.35.0		; <i32> [#uses=2]
	%tmp345 = trunc i32 %tmp343 to i8		; <i8> [#uses=1]
	%tmp348 = or i8 %tmp346, %tmp345		; <i8> [#uses=1]
	%tmp350 = load i8* %dest.3.0		; <i8> [#uses=1]
	%tmp351 = and i8 %tmp348, %tmp350		; <i8> [#uses=1]
	%tmp356 = and i32 %tmp354, %tmp343		; <i32> [#uses=1]
	%tmp356.upgrd.33 = trunc i32 %tmp356 to i8		; <i8> [#uses=1]
	%tmp357 = or i8 %tmp351, %tmp356.upgrd.33		; <i8> [#uses=1]
	store i8 %tmp357, i8* %dest.3.0
	%tmp362 = ptrtoint i8* %dest.3.0 to i32		; <i32> [#uses=1]
	%optr309.3.in51 = add i32 %tmp362, 1		; <i32> [#uses=2]
	%optr309.353 = inttoptr i32 %optr309.3.in51 to i8*		; <i8*> [#uses=2]
	br i1 %tmp39755, label %cond_true398.preheader, label %bb401

cond_true398.preheader:		; preds = %cond_next340
	br label %cond_true398

cond_true398:		; preds = %cond_true398, %cond_true398.preheader
	%indvar66 = phi i32 [ 0, %cond_true398.preheader ], [ %indvar.next67, %cond_true398 ]		; <i32> [#uses=4]
	%bptr307.4.0 = phi i8* [ %tmp370.upgrd.35, %cond_true398 ], [ %bptr307.3, %cond_true398.preheader ]		; <i8*> [#uses=2]
	%optr309.3.0 = phi i8* [ %optr309.3, %cond_true398 ], [ %optr309.353, %cond_true398.preheader ]		; <i8*> [#uses=2]
	%optr309.3.in.0 = add i32 %indvar66, %optr309.3.in51		; <i32> [#uses=1]
	%tmp.70 = add i32 %tmp109, %w		; <i32> [#uses=1]
	%tmp.72 = mul i32 %indvar66, -8		; <i32> [#uses=1]
	%tmp.71 = add i32 %tmp.70, -8		; <i32> [#uses=1]
	%count308.3.0 = add i32 %tmp.72, %tmp.71		; <i32> [#uses=1]
	%tmp366 = load i8* %bptr307.4.0		; <i8> [#uses=1]
	%tmp366.upgrd.34 = zext i8 %tmp366 to i32		; <i32> [#uses=1]
	%tmp369 = ptrtoint i8* %bptr307.4.0 to i32		; <i32> [#uses=1]
	%tmp370 = add i32 %tmp369, 1		; <i32> [#uses=1]
	%tmp370.upgrd.35 = inttoptr i32 %tmp370 to i8*		; <i8*> [#uses=3]
	%tmp372 = load i8* %tmp370.upgrd.35		; <i8> [#uses=1]
	%tmp372.upgrd.36 = zext i8 %tmp372 to i32		; <i32> [#uses=1]
	%shift.upgrd.37 = zext i8 %tmp319 to i32		; <i32> [#uses=1]
	%tmp374463 = lshr i32 %tmp372.upgrd.36, %shift.upgrd.37		; <i32> [#uses=1]
	%shift.upgrd.38 = zext i8 %tmp324 to i32		; <i32> [#uses=1]
	%tmp368 = shl i32 %tmp366.upgrd.34, %shift.upgrd.38		; <i32> [#uses=1]
	%tmp377 = add i32 %tmp374463, %tmp368		; <i32> [#uses=1]
	%tmp379 = xor i32 %tmp377, %iftmp.35.0		; <i32> [#uses=2]
	%tmp382 = or i32 %tmp379, %iftmp.36.0		; <i32> [#uses=1]
	%tmp382.upgrd.39 = trunc i32 %tmp382 to i8		; <i8> [#uses=1]
	%tmp384 = load i8* %optr309.3.0		; <i8> [#uses=1]
	%tmp385 = and i8 %tmp382.upgrd.39, %tmp384		; <i8> [#uses=1]
	%tmp388 = and i32 %tmp379, %iftmp.37.0		; <i32> [#uses=1]
	%tmp388.upgrd.40 = trunc i32 %tmp388 to i8		; <i8> [#uses=1]
	%tmp389 = or i8 %tmp385, %tmp388.upgrd.40		; <i8> [#uses=1]
	store i8 %tmp389, i8* %optr309.3.0
	%tmp392 = add i32 %count308.3.0, -8		; <i32> [#uses=2]
	%optr309.3.in = add i32 %optr309.3.in.0, 1		; <i32> [#uses=1]
	%optr309.3 = inttoptr i32 %optr309.3.in to i8*		; <i8*> [#uses=2]
	%tmp397 = icmp sgt i32 %tmp392, 7		; <i1> [#uses=1]
	%indvar.next67 = add i32 %indvar66, 1		; <i32> [#uses=1]
	br i1 %tmp397, label %cond_true398, label %bb401.loopexit

bb401.loopexit:		; preds = %cond_true398
	br label %bb401

bb401:		; preds = %bb401.loopexit, %cond_next340
	%count308.3.1 = phi i32 [ %tmp361, %cond_next340 ], [ %tmp392, %bb401.loopexit ]		; <i32> [#uses=2]
	%bptr307.4.1 = phi i8* [ %bptr307.3, %cond_next340 ], [ %tmp370.upgrd.35, %bb401.loopexit ]		; <i8*> [#uses=2]
	%optr309.3.1 = phi i8* [ %optr309.353, %cond_next340 ], [ %optr309.3, %bb401.loopexit ]		; <i8*> [#uses=2]
	%tmp403 = icmp sgt i32 %count308.3.1, 0		; <i1> [#uses=1]
	br i1 %tmp403, label %cond_true404, label %cond_next442

cond_true404:		; preds = %bb401
	%tmp406 = load i8* %bptr307.4.1		; <i8> [#uses=1]
	%tmp406.upgrd.41 = zext i8 %tmp406 to i32		; <i32> [#uses=1]
	%shift.upgrd.42 = zext i8 %tmp324 to i32		; <i32> [#uses=1]
	%tmp408 = shl i32 %tmp406.upgrd.41, %shift.upgrd.42		; <i32> [#uses=2]
	%tmp413 = icmp sgt i32 %count308.3.1, %tmp303		; <i1> [#uses=1]
	br i1 %tmp413, label %cond_true414, label %cond_next422

cond_true414:		; preds = %cond_true404
	%tmp409 = ptrtoint i8* %bptr307.4.1 to i32		; <i32> [#uses=1]
	%tmp410 = add i32 %tmp409, 1		; <i32> [#uses=1]
	%tmp410.upgrd.43 = inttoptr i32 %tmp410 to i8*		; <i8*> [#uses=1]
	%tmp416 = load i8* %tmp410.upgrd.43		; <i8> [#uses=1]
	%tmp416.upgrd.44 = zext i8 %tmp416 to i32		; <i32> [#uses=1]
	%shift.upgrd.45 = zext i8 %tmp319 to i32		; <i32> [#uses=1]
	%tmp418 = lshr i32 %tmp416.upgrd.44, %shift.upgrd.45		; <i32> [#uses=2]
	%tmp420 = add i32 %tmp418, %tmp408		; <i32> [#uses=1]
	br label %cond_next422

cond_next422:		; preds = %cond_true414, %cond_true404
	%bits.6 = phi i32 [ %tmp420, %cond_true414 ], [ %tmp408, %cond_true404 ]		; <i32> [#uses=1]
	%tmp425 = xor i32 %bits.6, %iftmp.35.0		; <i32> [#uses=1]
	%tmp427 = trunc i32 %tmp425 to i8		; <i8> [#uses=2]
	%tmp430 = or i8 %tmp428, %tmp427		; <i8> [#uses=1]
	%tmp432 = load i8* %optr309.3.1		; <i8> [#uses=1]
	%tmp433 = and i8 %tmp430, %tmp432		; <i8> [#uses=1]
	%tmp438 = and i8 %tmp436.upgrd.47, %tmp427		; <i8> [#uses=1]
	%tmp439 = or i8 %tmp433, %tmp438		; <i8> [#uses=1]
	store i8 %tmp439, i8* %optr309.3.1
	br label %cond_next442

cond_next442:		; preds = %cond_next422, %bb401
	%tmp443 = ptrtoint i8** %dest_line.3.0 to i32		; <i32> [#uses=1]
	%tmp444 = add i32 %tmp443, 4		; <i32> [#uses=1]
	%tmp444.upgrd.46 = inttoptr i32 %tmp444 to i8**		; <i8**> [#uses=2]
	%tmp446 = load i8** %tmp444.upgrd.46		; <i8*> [#uses=1]
	%tmp448 = getelementptr i8* %tmp446, i32 %tmp93		; <i8*> [#uses=1]
	%tmp454 = add i32 %tmp454.0, -1		; <i32> [#uses=1]
	%tmp456 = icmp slt i32 %tmp454, 0		; <i1> [#uses=1]
	%indvar.next75 = add i32 %indvar74, 1		; <i32> [#uses=1]
	br i1 %tmp456, label %return.loopexit56, label %cond_true457

cond_true457.preheader:		; preds = %cond_false299
	%tmp315 = icmp slt i32 %tmp107, %tmp110		; <i1> [#uses=1]
	%tmp319 = trunc i32 %tmp303 to i8		; <i8> [#uses=4]
	%tmp324 = trunc i32 %tmp305 to i8		; <i8> [#uses=3]
	%tmp330 = icmp slt i32 %tmp107, %w		; <i1> [#uses=1]
	%tmp344 = trunc i32 %mask.1.1 to i8		; <i8> [#uses=1]
	%tmp344not = xor i8 %tmp344, -1		; <i8> [#uses=1]
	%tmp347 = trunc i32 %iftmp.36.0 to i8		; <i8> [#uses=2]
	%tmp346 = or i8 %tmp347, %tmp344not		; <i8> [#uses=1]
	%tmp354 = and i32 %iftmp.37.0, %mask.1.1		; <i32> [#uses=1]
	%tmp361 = sub i32 %w, %tmp110		; <i32> [#uses=2]
	%tmp39755 = icmp sgt i32 %tmp361, 7		; <i1> [#uses=1]
	%tmp426 = trunc i32 %rmask.0.1 to i8		; <i8> [#uses=1]
	%tmp426not = xor i8 %tmp426, -1		; <i8> [#uses=1]
	%tmp428 = or i8 %tmp347, %tmp426not		; <i8> [#uses=1]
	%tmp436 = and i32 %iftmp.37.0, %rmask.0.1		; <i32> [#uses=1]
	%tmp436.upgrd.47 = trunc i32 %tmp436 to i8		; <i8> [#uses=1]
	br label %cond_true457

cond_true457:		; preds = %cond_true457.preheader, %cond_next442
	%indvar74 = phi i32 [ 0, %cond_true457.preheader ], [ %indvar.next75, %cond_next442 ]		; <i32> [#uses=3]
	%dest.3.0 = phi i8* [ %tmp448, %cond_next442 ], [ %tmp100, %cond_true457.preheader ]		; <i8*> [#uses=3]
	%dest_line.3.0 = phi i8** [ %tmp444.upgrd.46, %cond_next442 ], [ %tmp96, %cond_true457.preheader ]		; <i8**> [#uses=1]
	%tmp.77 = sub i32 0, %indvar74		; <i32> [#uses=2]
	%tmp454.0 = add i32 %tmp.77, %tmp45438		; <i32> [#uses=1]
	%tmp.79 = mul i32 %indvar74, %raster		; <i32> [#uses=2]
	%tmp104.sum = add i32 %tmp102, %tmp.79		; <i32> [#uses=1]
	%line.3.0 = getelementptr i8* %base, i32 %tmp104.sum		; <i8*> [#uses=3]
	%tmp318.upgrd.48 = load i8* %line.3.0		; <i8> [#uses=2]
	br i1 %tmp315, label %cond_false321, label %cond_true316

return.loopexit:		; preds = %cond_next280
	br label %return.loopexit.split

return.loopexit.split:		; preds = %return.loopexit, %return.loopexit.us
	br label %return

return.loopexit56:		; preds = %cond_next442
	br label %return

return:		; preds = %return.loopexit56, %return.loopexit.split, %cond_false299, %cond_true197, %cond_next78, %cond_next63, %bb58, %cond_next46
	%retval.0 = phi i32 [ 0, %cond_next46 ], [ -1, %bb58 ], [ -1, %cond_next63 ], [ -1, %cond_next78 ], [ 0, %cond_true197 ], [ 0, %cond_false299 ], [ 0, %return.loopexit.split ], [ 0, %return.loopexit56 ]		; <i32> [#uses=1]
	ret i32 %retval.0
}

declare i32 @mem_no_fault_proc(%struct.gx_device_memory*, i32, i32, i32, i32, i32)

declare i32 @mem_mono_fill_rectangle(%struct.gx_device*, i32, i32, i32, i32, i32)

declare i32 @mem_copy_mono_recover(%struct.gx_device*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32)
