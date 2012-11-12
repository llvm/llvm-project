; RUN: llc < %s -march=x86 | not grep imul

target triple = "i386-apple-darwin9.6"
	%struct.III_psy_xmin = type { [22 x double], [13 x [3 x double]] }
	%struct.III_scalefac_t = type { [22 x i32], [13 x [3 x i32]] }
	%struct.gr_info = type { i32, i32, i32, i32, i32, i32, i32, i32, [3 x i32], [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32*, [4 x i32] }
	%struct.lame_global_flags = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8*, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, float, float, float, float, i32, i32, i32, i32, i32, i32, i32, i32 }

define fastcc void @outer_loop(%struct.lame_global_flags* nocapture %gfp, double* nocapture %xr, i32 %targ_bits, double* nocapture %best_noise, %struct.III_psy_xmin* nocapture %l3_xmin, i32* nocapture %l3_enc, %struct.III_scalefac_t* nocapture %scalefac, %struct.gr_info* nocapture %cod_info, i32 %ch) nounwind {
entry:
	br label %bb4

bb4:		; preds = %bb4, %entry
	br i1 true, label %bb5, label %bb4

bb5:		; preds = %bb4
	br i1 true, label %bb28.i37, label %bb.i4

bb.i4:		; preds = %bb.i4, %bb5
	br label %bb.i4

bb28.i37:		; preds = %bb33.i47, %bb5
	%i.1.reg2mem.0.i = phi i32 [ %0, %bb33.i47 ], [ 0, %bb5 ]		; <i32> [#uses=2]
	%0 = add i32 %i.1.reg2mem.0.i, 1		; <i32> [#uses=2]
	br label %bb29.i38

bb29.i38:		; preds = %bb33.i47, %bb28.i37
	%indvar32.i = phi i32 [ %indvar.next33.i, %bb33.i47 ], [ 0, %bb28.i37 ]		; <i32> [#uses=2]
	%sfb.314.i = add i32 %indvar32.i, 0		; <i32> [#uses=3]
	%1 = getelementptr [4 x [21 x double]]* null, i32 0, i32 %0, i32 %sfb.314.i		; <double*> [#uses=1]
	%2 = load double* %1, align 8		; <double> [#uses=0]
	br i1 false, label %bb30.i41, label %bb33.i47

bb30.i41:		; preds = %bb29.i38
	%3 = getelementptr %struct.III_scalefac_t* null, i32 0, i32 1, i32 %sfb.314.i, i32 %i.1.reg2mem.0.i		; <i32*> [#uses=1]
	store i32 0, i32* %3, align 4
	br label %bb33.i47

bb33.i47:		; preds = %bb30.i41, %bb29.i38
	%4 = add i32 %sfb.314.i, 1		; <i32> [#uses=1]
	%phitmp.i46 = icmp ugt i32 %4, 11		; <i1> [#uses=1]
	%indvar.next33.i = add i32 %indvar32.i, 1		; <i32> [#uses=1]
	br i1 %phitmp.i46, label %bb28.i37, label %bb29.i38
}
