; RUN: llc < %s -mtriple=arm-apple-darwin

	%struct.H_TBL = type { [17 x i8], [256 x i8], i32 }
	%struct.Q_TBL = type { [64 x i16], i32 }
	%struct.anon = type { [80 x i8] }
	%struct.X_c_coef_ccler = type { void (%struct.X_Y*, i32)*, i32 (%struct.X_Y*, i8***)* }
	%struct.X_c_main_ccler = type { void (%struct.X_Y*, i32)*, void (%struct.X_Y*, i8**, i32*, i32)* }
	%struct.X_c_prep_ccler = type { void (%struct.X_Y*, i32)*, void (%struct.X_Y*, i8**, i32*, i32, i8***, i32*, i32)* }
	%struct.X_color_converter = type { void (%struct.X_Y*)*, void (%struct.X_Y*, i8**, i8***, i32, i32)* }
	%struct.X_common_struct = type { %struct.X_error_mgr*, %struct.X_memory_mgr*, %struct.X_progress_mgr*, i8*, i32, i32 }
	%struct.X_comp_main = type { void (%struct.X_Y*)*, void (%struct.X_Y*)*, void (%struct.X_Y*)*, i32, i32 }
	%struct.X_component_info = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.Q_TBL*, i8* }
	%struct.X_Y = type { %struct.X_error_mgr*, %struct.X_memory_mgr*, %struct.X_progress_mgr*, i8*, i32, i32, %struct.X_destination_mgr*, i32, i32, i32, i32, double, i32, i32, i32, %struct.X_component_info*, [4 x %struct.Q_TBL*], [4 x %struct.H_TBL*], [4 x %struct.H_TBL*], [16 x i8], [16 x i8], [16 x i8], i32, %struct.X_scan_info*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i16, i16, i32, i32, i32, i32, i32, i32, i32, [4 x %struct.X_component_info*], i32, i32, i32, [10 x i32], i32, i32, i32, i32, %struct.X_comp_main*, %struct.X_c_main_ccler*, %struct.X_c_prep_ccler*, %struct.X_c_coef_ccler*, %struct.X_marker_writer*, %struct.X_color_converter*, %struct.X_downssr*, %struct.X_forward_D*, %struct.X_entropy_en*, %struct.X_scan_info*, i32 }
	%struct.X_destination_mgr = type { i8*, i32, void (%struct.X_Y*)*, i32 (%struct.X_Y*)*, void (%struct.X_Y*)* }
	%struct.X_downssr = type { void (%struct.X_Y*)*, void (%struct.X_Y*, i8***, i32, i8***, i32)*, i32 }
	%struct.X_entropy_en = type { void (%struct.X_Y*, i32)*, i32 (%struct.X_Y*, [64 x i16]**)*, void (%struct.X_Y*)* }
	%struct.X_error_mgr = type { void (%struct.X_common_struct*)*, void (%struct.X_common_struct*, i32)*, void (%struct.X_common_struct*)*, void (%struct.X_common_struct*, i8*)*, void (%struct.X_common_struct*)*, i32, %struct.anon, i32, i32, i8**, i32, i8**, i32, i32 }
	%struct.X_forward_D = type { void (%struct.X_Y*)*, void (%struct.X_Y*, %struct.X_component_info*, i8**, [64 x i16]*, i32, i32, i32)* }
	%struct.X_marker_writer = type { void (%struct.X_Y*)*, void (%struct.X_Y*)*, void (%struct.X_Y*)*, void (%struct.X_Y*)*, void (%struct.X_Y*)*, void (%struct.X_Y*, i32, i32)*, void (%struct.X_Y*, i32)* }
	%struct.X_memory_mgr = type { i8* (%struct.X_common_struct*, i32, i32)*, i8* (%struct.X_common_struct*, i32, i32)*, i8** (%struct.X_common_struct*, i32, i32, i32)*, [64 x i16]** (%struct.X_common_struct*, i32, i32, i32)*, %struct.jvirt_sAY_cc* (%struct.X_common_struct*, i32, i32, i32, i32, i32)*, %struct.jvirt_bAY_cc* (%struct.X_common_struct*, i32, i32, i32, i32, i32)*, void (%struct.X_common_struct*)*, i8** (%struct.X_common_struct*, %struct.jvirt_sAY_cc*, i32, i32, i32)*, [64 x i16]** (%struct.X_common_struct*, %struct.jvirt_bAY_cc*, i32, i32, i32)*, void (%struct.X_common_struct*, i32)*, void (%struct.X_common_struct*)*, i32, i32 }
	%struct.X_progress_mgr = type { void (%struct.X_common_struct*)*, i32, i32, i32, i32 }
	%struct.X_scan_info = type { i32, [4 x i32], i32, i32, i32, i32 }
	%struct.jvirt_bAY_cc = type opaque
	%struct.jvirt_sAY_cc = type opaque

define void @test(%struct.X_Y* %cinfo) {
entry:
	br i1 false, label %bb.preheader, label %return

bb.preheader:		; preds = %entry
	%tbl.014.us = load i32, i32* null		; <i32> [#uses=1]
	br i1 false, label %cond_next.us, label %bb

cond_next51.us:		; preds = %cond_next.us, %cond_true33.us.cond_true46.us_crit_edge
	%htblptr.019.1.us = phi %struct.H_TBL** [ %tmp37.us, %cond_true33.us.cond_true46.us_crit_edge ], [ %tmp37.us, %cond_next.us ]		; <%struct.H_TBL**> [#uses=0]
	ret void

cond_true33.us.cond_true46.us_crit_edge:		; preds = %cond_next.us
	call void @_C_X_a_HT( )
	br label %cond_next51.us

cond_next.us:		; preds = %bb.preheader
	%tmp37.us = getelementptr %struct.X_Y, %struct.X_Y* %cinfo, i32 0, i32 17, i32 %tbl.014.us		; <%struct.H_TBL**> [#uses=3]
	%tmp4524.us = load %struct.H_TBL*, %struct.H_TBL** %tmp37.us		; <%struct.H_TBL*> [#uses=1]
	icmp eq %struct.H_TBL* %tmp4524.us, null		; <i1>:0 [#uses=1]
	br i1 %0, label %cond_true33.us.cond_true46.us_crit_edge, label %cond_next51.us

bb:		; preds = %bb.preheader
	ret void

return:		; preds = %entry
	ret void
}

declare void @_C_X_a_HT()
