; REQUIRES: asserts
; RUN: llc < %s -mtriple=i686-- -mcpu=yonah -stats 2>&1 | grep "Number of block tails merged" | grep 16
; PR1909

@.str = internal constant [48 x i8] c"transformed bounds: (%.2f, %.2f), (%.2f, %.2f)\0A\00"		; <ptr> [#uses=1]

define void @minmax(ptr %result) nounwind optsize {
entry:
	%tmp2 = load float, ptr %result, align 4		; <float> [#uses=6]
	%tmp4 = getelementptr float, ptr %result, i32 2		; <ptr> [#uses=5]
	%tmp5 = load float, ptr %tmp4, align 4		; <float> [#uses=10]
	%tmp7 = getelementptr float, ptr %result, i32 4		; <ptr> [#uses=5]
	%tmp8 = load float, ptr %tmp7, align 4		; <float> [#uses=8]
	%tmp10 = getelementptr float, ptr %result, i32 6		; <ptr> [#uses=3]
	%tmp11 = load float, ptr %tmp10, align 4		; <float> [#uses=8]
	%tmp12 = fcmp olt float %tmp8, %tmp11		; <i1> [#uses=5]
	br i1 %tmp12, label %bb, label %bb21

bb:		; preds = %entry
	%tmp23469 = fcmp olt float %tmp5, %tmp8		; <i1> [#uses=1]
	br i1 %tmp23469, label %bb26, label %bb30

bb21:		; preds = %entry
	%tmp23 = fcmp olt float %tmp5, %tmp11		; <i1> [#uses=1]
	br i1 %tmp23, label %bb26, label %bb30

bb26:		; preds = %bb21, %bb
	%tmp52471 = fcmp olt float %tmp2, %tmp5		; <i1> [#uses=1]
	br i1 %tmp52471, label %bb111, label %bb59

bb30:		; preds = %bb21, %bb
	br i1 %tmp12, label %bb40, label %bb50

bb40:		; preds = %bb30
	%tmp52473 = fcmp olt float %tmp2, %tmp8		; <i1> [#uses=1]
	br i1 %tmp52473, label %bb111, label %bb59

bb50:		; preds = %bb30
	%tmp52 = fcmp olt float %tmp2, %tmp11		; <i1> [#uses=1]
	br i1 %tmp52, label %bb111, label %bb59

bb59:		; preds = %bb50, %bb40, %bb26
	br i1 %tmp12, label %bb72, label %bb80

bb72:		; preds = %bb59
	%tmp82475 = fcmp olt float %tmp5, %tmp8		; <i1> [#uses=2]
	%brmerge786 = or i1 %tmp82475, %tmp12		; <i1> [#uses=1]
	%tmp4.mux787 = select i1 %tmp82475, ptr %tmp4, ptr %tmp7		; <ptr> [#uses=1]
	br i1 %brmerge786, label %bb111, label %bb103

bb80:		; preds = %bb59
	%tmp82 = fcmp olt float %tmp5, %tmp11		; <i1> [#uses=2]
	%brmerge = or i1 %tmp82, %tmp12		; <i1> [#uses=1]
	%tmp4.mux = select i1 %tmp82, ptr %tmp4, ptr %tmp7		; <ptr> [#uses=1]
	br i1 %brmerge, label %bb111, label %bb103

bb103:		; preds = %bb80, %bb72
	br label %bb111

bb111:		; preds = %bb103, %bb80, %bb72, %bb50, %bb40, %bb26
	%iftmp.0.0.in = phi ptr [ %tmp10, %bb103 ], [ %result, %bb26 ], [ %result, %bb40 ], [ %result, %bb50 ], [ %tmp4.mux, %bb80 ], [ %tmp4.mux787, %bb72 ]		; <ptr> [#uses=1]
	%iftmp.0.0 = load float, ptr %iftmp.0.0.in		; <float> [#uses=1]
	%tmp125 = fcmp ogt float %tmp8, %tmp11		; <i1> [#uses=5]
	br i1 %tmp125, label %bb128, label %bb136

bb128:		; preds = %bb111
	%tmp138477 = fcmp ogt float %tmp5, %tmp8		; <i1> [#uses=1]
	br i1 %tmp138477, label %bb141, label %bb145

bb136:		; preds = %bb111
	%tmp138 = fcmp ogt float %tmp5, %tmp11		; <i1> [#uses=1]
	br i1 %tmp138, label %bb141, label %bb145

bb141:		; preds = %bb136, %bb128
	%tmp167479 = fcmp ogt float %tmp2, %tmp5		; <i1> [#uses=1]
	br i1 %tmp167479, label %bb226, label %bb174

bb145:		; preds = %bb136, %bb128
	br i1 %tmp125, label %bb155, label %bb165

bb155:		; preds = %bb145
	%tmp167481 = fcmp ogt float %tmp2, %tmp8		; <i1> [#uses=1]
	br i1 %tmp167481, label %bb226, label %bb174

bb165:		; preds = %bb145
	%tmp167 = fcmp ogt float %tmp2, %tmp11		; <i1> [#uses=1]
	br i1 %tmp167, label %bb226, label %bb174

bb174:		; preds = %bb165, %bb155, %bb141
	br i1 %tmp125, label %bb187, label %bb195

bb187:		; preds = %bb174
	%tmp197483 = fcmp ogt float %tmp5, %tmp8		; <i1> [#uses=2]
	%brmerge790 = or i1 %tmp197483, %tmp125		; <i1> [#uses=1]
	%tmp4.mux791 = select i1 %tmp197483, ptr %tmp4, ptr %tmp7		; <ptr> [#uses=1]
	br i1 %brmerge790, label %bb226, label %bb218

bb195:		; preds = %bb174
	%tmp197 = fcmp ogt float %tmp5, %tmp11		; <i1> [#uses=2]
	%brmerge788 = or i1 %tmp197, %tmp125		; <i1> [#uses=1]
	%tmp4.mux789 = select i1 %tmp197, ptr %tmp4, ptr %tmp7		; <ptr> [#uses=1]
	br i1 %brmerge788, label %bb226, label %bb218

bb218:		; preds = %bb195, %bb187
	br label %bb226

bb226:		; preds = %bb218, %bb195, %bb187, %bb165, %bb155, %bb141
	%iftmp.7.0.in = phi ptr [ %tmp10, %bb218 ], [ %result, %bb141 ], [ %result, %bb155 ], [ %result, %bb165 ], [ %tmp4.mux789, %bb195 ], [ %tmp4.mux791, %bb187 ]		; <ptr> [#uses=1]
	%iftmp.7.0 = load float, ptr %iftmp.7.0.in		; <float> [#uses=1]
	%tmp229 = getelementptr float, ptr %result, i32 1		; <ptr> [#uses=7]
	%tmp230 = load float, ptr %tmp229, align 4		; <float> [#uses=6]
	%tmp232 = getelementptr float, ptr %result, i32 3		; <ptr> [#uses=5]
	%tmp233 = load float, ptr %tmp232, align 4		; <float> [#uses=10]
	%tmp235 = getelementptr float, ptr %result, i32 5		; <ptr> [#uses=5]
	%tmp236 = load float, ptr %tmp235, align 4		; <float> [#uses=8]
	%tmp238 = getelementptr float, ptr %result, i32 7		; <ptr> [#uses=3]
	%tmp239 = load float, ptr %tmp238, align 4		; <float> [#uses=8]
	%tmp240 = fcmp olt float %tmp236, %tmp239		; <i1> [#uses=5]
	br i1 %tmp240, label %bb243, label %bb251

bb243:		; preds = %bb226
	%tmp253485 = fcmp olt float %tmp233, %tmp236		; <i1> [#uses=1]
	br i1 %tmp253485, label %bb256, label %bb260

bb251:		; preds = %bb226
	%tmp253 = fcmp olt float %tmp233, %tmp239		; <i1> [#uses=1]
	br i1 %tmp253, label %bb256, label %bb260

bb256:		; preds = %bb251, %bb243
	%tmp282487 = fcmp olt float %tmp230, %tmp233		; <i1> [#uses=1]
	br i1 %tmp282487, label %bb341, label %bb289

bb260:		; preds = %bb251, %bb243
	br i1 %tmp240, label %bb270, label %bb280

bb270:		; preds = %bb260
	%tmp282489 = fcmp olt float %tmp230, %tmp236		; <i1> [#uses=1]
	br i1 %tmp282489, label %bb341, label %bb289

bb280:		; preds = %bb260
	%tmp282 = fcmp olt float %tmp230, %tmp239		; <i1> [#uses=1]
	br i1 %tmp282, label %bb341, label %bb289

bb289:		; preds = %bb280, %bb270, %bb256
	br i1 %tmp240, label %bb302, label %bb310

bb302:		; preds = %bb289
	%tmp312491 = fcmp olt float %tmp233, %tmp236		; <i1> [#uses=2]
	%brmerge793 = or i1 %tmp312491, %tmp240		; <i1> [#uses=1]
	%tmp232.mux794 = select i1 %tmp312491, ptr %tmp232, ptr %tmp235		; <ptr> [#uses=1]
	br i1 %brmerge793, label %bb341, label %bb333

bb310:		; preds = %bb289
	%tmp312 = fcmp olt float %tmp233, %tmp239		; <i1> [#uses=2]
	%brmerge792 = or i1 %tmp312, %tmp240		; <i1> [#uses=1]
	%tmp232.mux = select i1 %tmp312, ptr %tmp232, ptr %tmp235		; <ptr> [#uses=1]
	br i1 %brmerge792, label %bb341, label %bb333

bb333:		; preds = %bb310, %bb302
	br label %bb341

bb341:		; preds = %bb333, %bb310, %bb302, %bb280, %bb270, %bb256
	%iftmp.14.0.in = phi ptr [ %tmp238, %bb333 ], [ %tmp229, %bb280 ], [ %tmp229, %bb270 ], [ %tmp229, %bb256 ], [ %tmp232.mux, %bb310 ], [ %tmp232.mux794, %bb302 ]		; <ptr> [#uses=1]
	%iftmp.14.0 = load float, ptr %iftmp.14.0.in		; <float> [#uses=1]
	%tmp355 = fcmp ogt float %tmp236, %tmp239		; <i1> [#uses=5]
	br i1 %tmp355, label %bb358, label %bb366

bb358:		; preds = %bb341
	%tmp368493 = fcmp ogt float %tmp233, %tmp236		; <i1> [#uses=1]
	br i1 %tmp368493, label %bb371, label %bb375

bb366:		; preds = %bb341
	%tmp368 = fcmp ogt float %tmp233, %tmp239		; <i1> [#uses=1]
	br i1 %tmp368, label %bb371, label %bb375

bb371:		; preds = %bb366, %bb358
	%tmp397495 = fcmp ogt float %tmp230, %tmp233		; <i1> [#uses=1]
	br i1 %tmp397495, label %bb456, label %bb404

bb375:		; preds = %bb366, %bb358
	br i1 %tmp355, label %bb385, label %bb395

bb385:		; preds = %bb375
	%tmp397497 = fcmp ogt float %tmp230, %tmp236		; <i1> [#uses=1]
	br i1 %tmp397497, label %bb456, label %bb404

bb395:		; preds = %bb375
	%tmp397 = fcmp ogt float %tmp230, %tmp239		; <i1> [#uses=1]
	br i1 %tmp397, label %bb456, label %bb404

bb404:		; preds = %bb395, %bb385, %bb371
	br i1 %tmp355, label %bb417, label %bb425

bb417:		; preds = %bb404
	%tmp427499 = fcmp ogt float %tmp233, %tmp236		; <i1> [#uses=2]
	%brmerge797 = or i1 %tmp427499, %tmp355		; <i1> [#uses=1]
	%tmp232.mux798 = select i1 %tmp427499, ptr %tmp232, ptr %tmp235		; <ptr> [#uses=1]
	br i1 %brmerge797, label %bb456, label %bb448

bb425:		; preds = %bb404
	%tmp427 = fcmp ogt float %tmp233, %tmp239		; <i1> [#uses=2]
	%brmerge795 = or i1 %tmp427, %tmp355		; <i1> [#uses=1]
	%tmp232.mux796 = select i1 %tmp427, ptr %tmp232, ptr %tmp235		; <ptr> [#uses=1]
	br i1 %brmerge795, label %bb456, label %bb448

bb448:		; preds = %bb425, %bb417
	br label %bb456

bb456:		; preds = %bb448, %bb425, %bb417, %bb395, %bb385, %bb371
	%iftmp.21.0.in = phi ptr [ %tmp238, %bb448 ], [ %tmp229, %bb395 ], [ %tmp229, %bb385 ], [ %tmp229, %bb371 ], [ %tmp232.mux796, %bb425 ], [ %tmp232.mux798, %bb417 ]		; <ptr> [#uses=1]
	%iftmp.21.0 = load float, ptr %iftmp.21.0.in		; <float> [#uses=1]
	%tmp458459 = fpext float %iftmp.21.0 to double		; <double> [#uses=1]
	%tmp460461 = fpext float %iftmp.7.0 to double		; <double> [#uses=1]
	%tmp462463 = fpext float %iftmp.14.0 to double		; <double> [#uses=1]
	%tmp464465 = fpext float %iftmp.0.0 to double		; <double> [#uses=1]
	%tmp467 = tail call i32 (ptr, ...) @printf( ptr @.str, double %tmp464465, double %tmp462463, double %tmp460461, double %tmp458459 ) nounwind 		; <i32> [#uses=0]
	ret void
}

declare i32 @printf(ptr, ...) nounwind
