; RUN: opt < %s -passes=newgvn -S | FileCheck %s

@img_width = external global i16		; <ptr> [#uses=2]

define i32 @smpUMHEXBipredIntegerPelBlockMotionSearch(ptr %cur_pic, i16 signext  %ref, i32 %list, i32 %pic_pix_x, i32 %pic_pix_y, i32 %blocktype, i16 signext  %pred_mv_x1, i16 signext  %pred_mv_y1, i16 signext  %pred_mv_x2, i16 signext  %pred_mv_y2, ptr %mv_x, ptr %mv_y, ptr %s_mv_x, ptr %s_mv_y, i32 %search_range, i32 %min_mcost, i32 %lambda_factor) {
cond_next143:		; preds = %entry
	store i16 0, ptr @img_width, align 2
	br i1 false, label %cond_next449, label %cond_false434

cond_false434:		; preds = %cond_true415
	br label %cond_next449

cond_next449:		; preds = %cond_false434, %cond_true415
	br i1 false, label %cond_next698, label %cond_false470

cond_false470:		; preds = %cond_next449
	br label %cond_next698

cond_next698:		; preds = %cond_true492
	%tmp701 = load i16, ptr @img_width, align 2		; <i16> [#uses=0]
; CHECK-NOT: %tmp701 =
	ret i32 0
}
