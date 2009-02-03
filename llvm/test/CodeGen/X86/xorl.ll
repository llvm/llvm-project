; RUN: llvm-as < %s | llc -march=x86 | grep xorl | count 1

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
	%struct.block_symbol = type { [3 x %struct.cgraph_rtl_info], %struct.object_block*, i64 }
	%struct.rtx_def = type <{ i16, i8, i8, %struct.u }>
	%struct.u = type { %struct.block_symbol }
	%struct.cgraph_rtl_info = type { i32 }
	%struct.object_block = type { %struct.section*, i32, i64, %struct.VEC_rtx_gc*, %struct.VEC_rtx_gc* }
	%struct.section = type { %struct.unnamed_section }
	%struct.VEC_rtx_base = type { i32, i32, [1 x %struct.rtx_def*] }
	%struct.VEC_rtx_gc = type { %struct.VEC_rtx_base }
	%struct.tree_common = type <{ %struct.tree_node*, %struct.tree_node*, %union.tree_ann_d*, i8, i8, i8, i8, i8, [3 x i8] }>
	%struct.tree_complex = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_node = type { %struct.tree_complex, [116 x i8] }
	%struct.unnamed_section = type { %struct.cgraph_rtl_info, void (i8*)*, i8*, %struct.section* }
	%union.tree_ann_d = type opaque

define %struct.rtx_def* @expand_call() nounwind  {
entry:
	br i1 false, label %bb216, label %bb171
bb171:		; preds = %entry
	ret %struct.rtx_def* null
bb216:		; preds = %entry
	br i1 false, label %bb336, label %bb222
bb222:		; preds = %bb216
	ret %struct.rtx_def* null
bb336:		; preds = %bb216
	br i1 false, label %bb429, label %bb417
bb417:		; preds = %bb336
	ret %struct.rtx_def* null
bb429:		; preds = %bb336
	br i1 false, label %bb713, label %bb493
bb493:		; preds = %bb429
	ret %struct.rtx_def* null
bb713:		; preds = %bb429
	br i1 false, label %bb810, label %bb797
bb797:		; preds = %bb713
	ret %struct.rtx_def* null
bb810:		; preds = %bb713
	br i1 false, label %bb822, label %bb815
bb815:		; preds = %bb810
	ret %struct.rtx_def* null
bb822:		; preds = %bb810
	br label %bb1652.preheader
bb919:		; preds = %bb1652.preheader
	ret %struct.rtx_def* null
bb1657:		; preds = %bb1652.preheader
	br i1 false, label %bb1666, label %bb1652.preheader
bb1652.preheader:		; preds = %bb1657, %bb822
	br i1 false, label %bb1657, label %bb919
bb1666:		; preds = %bb1657
	br i1 false, label %bb1815.preheader, label %bb1870
bb1815.preheader:		; preds = %bb1666
	br i1 false, label %bb1693, label %bb1828
bb1693:		; preds = %bb1815.preheader
	br i1 false, label %bb1718, label %bb1703
bb1703:		; preds = %bb1693
	ret %struct.rtx_def* null
bb1718:		; preds = %bb1693
	br i1 false, label %bb1741, label %bb1828
bb1741:		; preds = %bb1718
	switch i8 0, label %bb1775 [
		 i8 54, label %bb1798
		 i8 58, label %bb1798
		 i8 55, label %bb1798
	]
bb1775:		; preds = %bb1741
	ret %struct.rtx_def* null
bb1798:		; preds = %bb1741, %bb1741, %bb1741
	%tmp1811 = add i32 0, 0		; <i32> [#uses=1]
	br label %bb1828
bb1828:		; preds = %bb1798, %bb1718, %bb1815.preheader
	%copy_to_evaluate_size.1.lcssa = phi i32 [ 0, %bb1815.preheader ], [ %tmp1811, %bb1798 ], [ 0, %bb1718 ]		; <i32> [#uses=1]
	%tmp1830 = shl i32 %copy_to_evaluate_size.1.lcssa, 1		; <i32> [#uses=1]
	%tmp18301831 = sext i32 %tmp1830 to i64		; <i64> [#uses=1]
	%tmp1835 = icmp slt i64 %tmp18301831, 0		; <i1> [#uses=1]
	%tmp1835.not = xor i1 %tmp1835, true		; <i1> [#uses=1]
	%bothcond6193 = and i1 %tmp1835.not, false		; <i1> [#uses=1]
	br i1 %bothcond6193, label %bb1845, label %bb1870
bb1845:		; preds = %bb1828
	ret %struct.rtx_def* null
bb1870:		; preds = %bb1828, %bb1666
	ret %struct.rtx_def* null
}
