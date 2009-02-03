; PR1495
; RUN: llvm-as < %s | llc -march=x86

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-pc-linux-gnu"
	%struct.AVRational = type { i32, i32 }
	%struct.FFTComplex = type { float, float }
	%struct.FFTContext = type { i32, i32, i16*, %struct.FFTComplex*, %struct.FFTComplex*, void (%struct.FFTContext*, %struct.FFTComplex*)*, void (%struct.MDCTContext*, float*, float*, float*)* }
	%struct.MDCTContext = type { i32, i32, float*, float*, %struct.FFTContext }
	%struct.Minima = type { i32, i32, i32, i32 }
	%struct.codebook_t = type { i32, i8*, i32*, i32, float, float, i32, i32, i32*, float*, float* }
	%struct.floor_class_t = type { i32, i32, i32, i32* }
	%struct.floor_t = type { i32, i32*, i32, %struct.floor_class_t*, i32, i32, i32, %struct.Minima* }
	%struct.mapping_t = type { i32, i32*, i32*, i32*, i32, i32*, i32* }
	%struct.residue_t = type { i32, i32, i32, i32, i32, i32, [8 x i8]*, [2 x float]* }
	%struct.venc_context_t = type { i32, i32, [2 x i32], [2 x %struct.MDCTContext], [2 x float*], i32, float*, float*, float*, float*, float, i32, %struct.codebook_t*, i32, %struct.floor_t*, i32, %struct.residue_t*, i32, %struct.mapping_t*, i32, %struct.AVRational* }

define fastcc i32 @put_main_header(%struct.venc_context_t* %venc, i8** %out) {
entry:
	br i1 false, label %bb1820, label %bb288.bb148_crit_edge

bb288.bb148_crit_edge:		; preds = %entry
	ret i32 0

cond_next1712:		; preds = %bb1820.bb1680_crit_edge
	ret i32 0

bb1817:		; preds = %bb1820.bb1680_crit_edge
	br label %bb1820

bb1820:		; preds = %bb1817, %entry
	%pb.1.50 = phi i32 [ %tmp1693, %bb1817 ], [ 8, %entry ]		; <i32> [#uses=3]
	br i1 false, label %bb2093, label %bb1820.bb1680_crit_edge

bb1820.bb1680_crit_edge:		; preds = %bb1820
	%tmp1693 = add i32 %pb.1.50, 8		; <i32> [#uses=2]
	%tmp1702 = icmp slt i32 %tmp1693, 0		; <i1> [#uses=1]
	br i1 %tmp1702, label %cond_next1712, label %bb1817

bb2093:		; preds = %bb1820
	%tmp2102 = add i32 %pb.1.50, 65		; <i32> [#uses=0]
	%tmp2236 = add i32 %pb.1.50, 72		; <i32> [#uses=1]
	%tmp2237 = sdiv i32 %tmp2236, 8		; <i32> [#uses=2]
	br i1 false, label %bb2543, label %bb2536.bb2396_crit_edge

bb2536.bb2396_crit_edge:		; preds = %bb2093
	ret i32 0

bb2543:		; preds = %bb2093
	br i1 false, label %cond_next2576, label %bb2690

cond_next2576:		; preds = %bb2543
	ret i32 0

bb2682:		; preds = %bb2690
	ret i32 0

bb2690:		; preds = %bb2543
	br i1 false, label %bb2682, label %bb2698

bb2698:		; preds = %bb2690
	br i1 false, label %cond_next2726, label %bb2831

cond_next2726:		; preds = %bb2698
	ret i32 0

bb2831:		; preds = %bb2698
	br i1 false, label %cond_next2859, label %bb2964

cond_next2859:		; preds = %bb2831
	br i1 false, label %bb2943, label %cond_true2866

cond_true2866:		; preds = %cond_next2859
	br i1 false, label %cond_true2874, label %cond_false2897

cond_true2874:		; preds = %cond_true2866
	ret i32 0

cond_false2897:		; preds = %cond_true2866
	ret i32 0

bb2943:		; preds = %cond_next2859
	ret i32 0

bb2964:		; preds = %bb2831
	br i1 false, label %cond_next2997, label %bb4589

cond_next2997:		; preds = %bb2964
	ret i32 0

bb3103:		; preds = %bb4589
	ret i32 0

bb4589:		; preds = %bb2964
	br i1 false, label %bb3103, label %bb4597

bb4597:		; preds = %bb4589
	br i1 false, label %cond_next4630, label %bb4744

cond_next4630:		; preds = %bb4597
	br i1 false, label %bb4744, label %cond_true4724

cond_true4724:		; preds = %cond_next4630
	br i1 false, label %bb4736, label %bb7531

bb4736:		; preds = %cond_true4724
	ret i32 0

bb4744:		; preds = %cond_next4630, %bb4597
	ret i32 0

bb7531:		; preds = %cond_true4724
	%v_addr.023.0.i6 = add i32 %tmp2237, -255		; <i32> [#uses=1]
	br label %bb.i14

bb.i14:		; preds = %bb.i14, %bb7531
	%n.021.0.i8 = phi i32 [ 0, %bb7531 ], [ %indvar.next, %bb.i14 ]		; <i32> [#uses=2]
	%tmp..i9 = mul i32 %n.021.0.i8, -255		; <i32> [#uses=1]
	%tmp5.i11 = add i32 %v_addr.023.0.i6, %tmp..i9		; <i32> [#uses=1]
	%tmp10.i12 = icmp ugt i32 %tmp5.i11, 254		; <i1> [#uses=1]
	%indvar.next = add i32 %n.021.0.i8, 1		; <i32> [#uses=1]
	br i1 %tmp10.i12, label %bb.i14, label %bb12.loopexit.i18

bb12.loopexit.i18:		; preds = %bb.i14
	call void @llvm.memcpy.i32( i8* null, i8* null, i32 %tmp2237, i32 1 )
	ret i32 0
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)
