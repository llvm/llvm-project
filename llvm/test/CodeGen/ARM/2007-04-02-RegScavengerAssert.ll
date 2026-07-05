; RUN: llc < %s -mtriple=arm-apple-darwin

	%struct.H_TBL = type { [17 x i8], [256 x i8], i32 }
	%struct.Q_TBL = type { [64 x i16], i32 }
	%struct.anon = type { [80 x i8] }
	%struct.X_c_coef_ccler = type { ptr, ptr }
	%struct.X_c_main_ccler = type { ptr, ptr }
	%struct.X_c_prep_ccler = type { ptr, ptr }
	%struct.X_color_converter = type { ptr, ptr }
	%struct.X_common_struct = type { ptr, ptr, ptr, ptr, i32, i32 }
	%struct.X_comp_main = type { ptr, ptr, ptr, i32, i32 }
	%struct.X_component_info = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr }
	%struct.X_Y = type { ptr, ptr, ptr, ptr, i32, i32, ptr, i32, i32, i32, i32, double, i32, i32, i32, ptr, [4 x ptr], [4 x ptr], [4 x ptr], [16 x i8], [16 x i8], [16 x i8], i32, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i16, i16, i32, i32, i32, i32, i32, i32, i32, [4 x ptr], i32, i32, i32, [10 x i32], i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32 }
	%struct.X_destination_mgr = type { ptr, i32, ptr, ptr, ptr }
	%struct.X_downssr = type { ptr, ptr, i32 }
	%struct.X_entropy_en = type { ptr, ptr, ptr }
	%struct.X_error_mgr = type { ptr, ptr, ptr, ptr, ptr, i32, %struct.anon, i32, i32, ptr, i32, ptr, i32, i32 }
	%struct.X_forward_D = type { ptr, ptr }
	%struct.X_marker_writer = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr }
	%struct.X_memory_mgr = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32 }
	%struct.X_progress_mgr = type { ptr, i32, i32, i32, i32 }
	%struct.X_scan_info = type { i32, [4 x i32], i32, i32, i32, i32 }
	%struct.jvirt_bAY_cc = type opaque
	%struct.jvirt_sAY_cc = type opaque

define void @test(ptr %cinfo) {
entry:
	br i1 false, label %bb.preheader, label %return

bb.preheader:		; preds = %entry
	%tbl.014.us = load i32, ptr null		; <i32> [#uses=1]
	br i1 false, label %cond_next.us, label %bb

cond_next51.us:		; preds = %cond_next.us, %cond_true33.us.cond_true46.us_crit_edge
	%htblptr.019.1.us = phi ptr [ %tmp37.us, %cond_true33.us.cond_true46.us_crit_edge ], [ %tmp37.us, %cond_next.us ]		; <ptr> [#uses=0]
	ret void

cond_true33.us.cond_true46.us_crit_edge:		; preds = %cond_next.us
	call void @_C_X_a_HT( )
	br label %cond_next51.us

cond_next.us:		; preds = %bb.preheader
	%tmp37.us = getelementptr %struct.X_Y, ptr %cinfo, i32 0, i32 17, i32 %tbl.014.us		; <ptr> [#uses=3]
	%tmp4524.us = load ptr, ptr %tmp37.us		; <ptr> [#uses=1]
	icmp eq ptr %tmp4524.us, null		; <i1>:0 [#uses=1]
	br i1 %0, label %cond_true33.us.cond_true46.us_crit_edge, label %cond_next51.us

bb:		; preds = %bb.preheader
	ret void

return:		; preds = %entry
	ret void
}

declare void @_C_X_a_HT()
