; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin | grep movl | not grep rax

	%struct.color_sample = type { i64 }
	%struct.gs_matrix = type { float, i64, float, i64, float, i64, float, i64, float, i64, float, i64 }
	%struct.ref = type { %struct.color_sample, i16, i16 }
	%struct.status = type { %struct.gs_matrix, i8*, i32, i32, i8*, i32, i32, i32, i32, i32, i32, i32 }

define i32 @ztype1imagepath(%struct.ref* %op) {
entry:
	br i1 false, label %cond_next, label %UnifiedReturnBlock

cond_next:		; preds = %entry
	br i1 false, label %cond_next68, label %UnifiedReturnBlock

cond_next68:		; preds = %cond_next
	%tmp5.i.i = malloc i8, i32 0		; <i8*> [#uses=2]
	br i1 false, label %bb81.outer.i, label %xit.i

bb81.outer.i:		; preds = %bb87.i, %cond_next68
	%tmp67.i = add i32 0, 1		; <i32> [#uses=1]
	br label %bb81.i

bb61.i:		; preds = %bb81.i
	%tmp71.i = getelementptr i8* %tmp5.i.i, i64 0		; <i8*> [#uses=1]
	%tmp72.i = load i8* %tmp71.i, align 1		; <i8> [#uses=1]
	%tmp73.i = icmp eq i8 %tmp72.i, 0		; <i1> [#uses=1]
	br i1 %tmp73.i, label %bb81.i, label %xit.i

bb81.i:		; preds = %bb61.i, %bb81.outer.i
	br i1 false, label %bb87.i, label %bb61.i

bb87.i:		; preds = %bb81.i
	br i1 false, label %bb81.outer.i, label %xit.i

xit.i:		; preds = %bb87.i, %bb61.i, %cond_next68
	%lsbx.0.reg2mem.1.i = phi i32 [ 0, %cond_next68 ], [ 0, %bb61.i ], [ %tmp67.i, %bb87.i ]		; <i32> [#uses=1]
	%tmp6162.i.i = fptrunc double 0.000000e+00 to float		; <float> [#uses=1]
	%tmp67.i15.i = fptrunc double 0.000000e+00 to float		; <float> [#uses=1]
	%tmp24.i27.i = icmp eq i64 0, 0		; <i1> [#uses=1]
	br i1 %tmp24.i27.i, label %cond_next.i79.i, label %cond_true.i34.i

cond_true.i34.i:		; preds = %xit.i
	ret i32 0

cond_next.i79.i:		; preds = %xit.i
	%phitmp167.i = fptosi double 0.000000e+00 to i64		; <i64> [#uses=1]
	%tmp142143.i = fpext float %tmp6162.i.i to double		; <double> [#uses=1]
	%tmp2.i139.i = add double %tmp142143.i, 5.000000e-01		; <double> [#uses=1]
	%tmp23.i140.i = fptosi double %tmp2.i139.i to i64		; <i64> [#uses=1]
	br i1 false, label %cond_true.i143.i, label %round_coord.exit148.i

cond_true.i143.i:		; preds = %cond_next.i79.i
	%tmp8.i142.i = icmp sgt i64 %tmp23.i140.i, -32768		; <i1> [#uses=1]
	br i1 %tmp8.i142.i, label %cond_true11.i145.i, label %round_coord.exit148.i

cond_true11.i145.i:		; preds = %cond_true.i143.i
	ret i32 0

round_coord.exit148.i:		; preds = %cond_true.i143.i, %cond_next.i79.i
	%tmp144149.i = phi i32 [ 32767, %cond_next.i79.i ], [ -32767, %cond_true.i143.i ]		; <i32> [#uses=1]
	store i32 %tmp144149.i, i32* null, align 8
	%tmp147148.i = fpext float %tmp67.i15.i to double		; <double> [#uses=1]
	%tmp2.i128.i = add double %tmp147148.i, 5.000000e-01		; <double> [#uses=1]
	%tmp23.i129.i = fptosi double %tmp2.i128.i to i64		; <i64> [#uses=2]
	%tmp5.i130.i = icmp slt i64 %tmp23.i129.i, 32768		; <i1> [#uses=1]
	br i1 %tmp5.i130.i, label %cond_true.i132.i, label %round_coord.exit137.i

cond_true.i132.i:		; preds = %round_coord.exit148.i
	%tmp8.i131.i = icmp sgt i64 %tmp23.i129.i, -32768		; <i1> [#uses=1]
	br i1 %tmp8.i131.i, label %cond_true11.i134.i, label %round_coord.exit137.i

cond_true11.i134.i:		; preds = %cond_true.i132.i
	br label %round_coord.exit137.i

round_coord.exit137.i:		; preds = %cond_true11.i134.i, %cond_true.i132.i, %round_coord.exit148.i
	%tmp149138.i = phi i32 [ 0, %cond_true11.i134.i ], [ 32767, %round_coord.exit148.i ], [ -32767, %cond_true.i132.i ]		; <i32> [#uses=1]
	br i1 false, label %cond_true.i121.i, label %round_coord.exit126.i

cond_true.i121.i:		; preds = %round_coord.exit137.i
	br i1 false, label %cond_true11.i123.i, label %round_coord.exit126.i

cond_true11.i123.i:		; preds = %cond_true.i121.i
	br label %round_coord.exit126.i

round_coord.exit126.i:		; preds = %cond_true11.i123.i, %cond_true.i121.i, %round_coord.exit137.i
	%tmp153127.i = phi i32 [ 0, %cond_true11.i123.i ], [ 32767, %round_coord.exit137.i ], [ -32767, %cond_true.i121.i ]		; <i32> [#uses=1]
	br i1 false, label %cond_true.i110.i, label %round_coord.exit115.i

cond_true.i110.i:		; preds = %round_coord.exit126.i
	br i1 false, label %cond_true11.i112.i, label %round_coord.exit115.i

cond_true11.i112.i:		; preds = %cond_true.i110.i
	br label %round_coord.exit115.i

round_coord.exit115.i:		; preds = %cond_true11.i112.i, %cond_true.i110.i, %round_coord.exit126.i
	%tmp157116.i = phi i32 [ 0, %cond_true11.i112.i ], [ 32767, %round_coord.exit126.i ], [ -32767, %cond_true.i110.i ]		; <i32> [#uses=2]
	br i1 false, label %cond_true.i99.i, label %round_coord.exit104.i

cond_true.i99.i:		; preds = %round_coord.exit115.i
	br i1 false, label %cond_true11.i101.i, label %round_coord.exit104.i

cond_true11.i101.i:		; preds = %cond_true.i99.i
	%tmp1213.i100.i = trunc i64 %phitmp167.i to i32		; <i32> [#uses=1]
	br label %cond_next172.i

round_coord.exit104.i:		; preds = %cond_true.i99.i, %round_coord.exit115.i
	%UnifiedRetVal.i102.i = phi i32 [ 32767, %round_coord.exit115.i ], [ -32767, %cond_true.i99.i ]		; <i32> [#uses=1]
	%tmp164.i = call fastcc i32 @put_int( %struct.status* null, i32 %tmp157116.i )		; <i32> [#uses=0]
	br label %cond_next172.i

cond_next172.i:		; preds = %round_coord.exit104.i, %cond_true11.i101.i
	%tmp161105.reg2mem.0.i = phi i32 [ %tmp1213.i100.i, %cond_true11.i101.i ], [ %UnifiedRetVal.i102.i, %round_coord.exit104.i ]		; <i32> [#uses=1]
	%tmp174.i = icmp eq i32 %tmp153127.i, 0		; <i1> [#uses=1]
	%bothcond.i = and i1 false, %tmp174.i		; <i1> [#uses=1]
	%tmp235.i = call fastcc i32 @put_int( %struct.status* null, i32 %tmp149138.i )		; <i32> [#uses=0]
	%tmp245.i = load i8** null, align 8		; <i8*> [#uses=2]
	%tmp246.i = getelementptr i8* %tmp245.i, i64 1		; <i8*> [#uses=1]
	br i1 %bothcond.i, label %cond_next254.i, label %bb259.i

cond_next254.i:		; preds = %cond_next172.i
	store i8 13, i8* %tmp245.i, align 1
	br label %bb259.i

bb259.i:		; preds = %cond_next254.i, %cond_next172.i
	%storemerge.i = phi i8* [ %tmp246.i, %cond_next254.i ], [ null, %cond_next172.i ]		; <i8*> [#uses=0]
	%tmp261.i = shl i32 %lsbx.0.reg2mem.1.i, 2		; <i32> [#uses=1]
	store i32 %tmp261.i, i32* null, align 8
	%tmp270.i = add i32 0, %tmp157116.i		; <i32> [#uses=1]
	store i32 %tmp270.i, i32* null, align 8
	%tmp275.i = add i32 0, %tmp161105.reg2mem.0.i		; <i32> [#uses=0]
	br i1 false, label %trace_cells.exit.i, label %bb.preheader.i.i

bb.preheader.i.i:		; preds = %bb259.i
	ret i32 0

trace_cells.exit.i:		; preds = %bb259.i
	free i8* %tmp5.i.i
	ret i32 0

UnifiedReturnBlock:		; preds = %cond_next, %entry
	ret i32 -20
}

declare fastcc i32 @put_int(%struct.status*, i32)
