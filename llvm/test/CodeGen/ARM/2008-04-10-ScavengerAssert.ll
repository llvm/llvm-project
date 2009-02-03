; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin

	%struct.CONTENTBOX = type { i32, i32, i32, i32, i32 }
	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.LOCBOX = type { i32, i32, i32, i32 }
	%struct.SIDEBOX = type { i32, i32 }
	%struct.UNCOMBOX = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
	%struct.cellbox = type { i8*, i32, i32, i32, [9 x i32], i32, i32, i32, i32, i32, i32, i32, double, double, double, double, double, i32, i32, %struct.CONTENTBOX*, %struct.UNCOMBOX*, [8 x %struct.tilebox*], %struct.SIDEBOX* }
	%struct.termbox = type { %struct.termbox*, i32, i32, i32, i32, i32 }
	%struct.tilebox = type { %struct.tilebox*, double, double, double, double, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.termbox*, %struct.LOCBOX* }
@.str127 = external constant [2 x i8]		; <[2 x i8]*> [#uses=1]
@.str584 = external constant [5 x i8]		; <[5 x i8]*> [#uses=1]
@.str8115 = external constant [9 x i8]		; <[9 x i8]*> [#uses=1]

declare %struct.FILE* @fopen(i8*, i8*)

declare i32 @strcmp(i8*, i8*)

declare i32 @fscanf(%struct.FILE*, i8*, ...)

define void @main(i32 %argc, i8** %argv) noreturn  {
entry:
	br i1 false, label %cond_next48, label %cond_false674
cond_next48:		; preds = %entry
	%tmp61 = call %struct.FILE* @fopen( i8* null, i8* getelementptr ([2 x i8]* @.str127, i32 0, i32 0) )		; <%struct.FILE*> [#uses=2]
	br i1 false, label %bb220.i.i.i, label %bb62.preheader.i.i.i
bb62.preheader.i.i.i:		; preds = %cond_next48
	ret void
bb220.i.i.i:		; preds = %cond_next48
	br i1 false, label %bb248.i.i.i, label %cond_next232.i.i.i
cond_next232.i.i.i:		; preds = %bb220.i.i.i
	ret void
bb248.i.i.i:		; preds = %bb220.i.i.i
	br i1 false, label %bb300.i.i.i, label %cond_false256.i.i.i
cond_false256.i.i.i:		; preds = %bb248.i.i.i
	ret void
bb300.i.i.i:		; preds = %bb248.i.i.i
	br label %bb.i.i347.i
bb.i.i347.i:		; preds = %bb.i.i347.i, %bb300.i.i.i
	br i1 false, label %bb894.loopexit.i.i, label %bb.i.i347.i
bb.i350.i:		; preds = %bb894.i.i
	br i1 false, label %bb24.i.i, label %cond_false373.i.i
bb24.i.i:		; preds = %bb24.i.i, %bb.i350.i
	br i1 false, label %bb40.i.i, label %bb24.i.i
bb40.i.i:		; preds = %bb24.i.i
	br i1 false, label %bb177.i393.i, label %bb82.i.i
bb82.i.i:		; preds = %bb40.i.i
	ret void
bb177.i393.i:		; preds = %bb40.i.i
	br i1 false, label %bb894.i.i, label %bb192.i.i
bb192.i.i:		; preds = %bb177.i393.i
	ret void
cond_false373.i.i:		; preds = %bb.i350.i
	%tmp376.i.i = call i32 @strcmp( i8* null, i8* getelementptr ([9 x i8]* @.str8115, i32 0, i32 0) )		; <i32> [#uses=0]
	br i1 false, label %cond_true380.i.i, label %cond_next602.i.i
cond_true380.i.i:		; preds = %cond_false373.i.i
	%tmp394.i418.i = add i32 %cell.0.i.i, 1		; <i32> [#uses=1]
	%tmp397.i420.i = load %struct.cellbox** null, align 4		; <%struct.cellbox*> [#uses=1]
	br label %bb398.i.i
bb398.i.i:		; preds = %bb398.i.i, %cond_true380.i.i
	br i1 false, label %bb414.i.i, label %bb398.i.i
bb414.i.i:		; preds = %bb398.i.i
	br i1 false, label %bb581.i.i, label %bb455.i442.i
bb455.i442.i:		; preds = %bb414.i.i
	ret void
bb581.i.i:		; preds = %bb581.i.i, %bb414.i.i
	br i1 false, label %bb894.i.i, label %bb581.i.i
cond_next602.i.i:		; preds = %cond_false373.i.i
	br i1 false, label %bb609.i.i, label %bb661.i.i
bb609.i.i:		; preds = %cond_next602.i.i
	br label %bb620.i.i
bb620.i.i:		; preds = %bb620.i.i, %bb609.i.i
	%indvar166.i465.i = phi i32 [ %indvar.next167.i.i, %bb620.i.i ], [ 0, %bb609.i.i ]		; <i32> [#uses=1]
	%tmp640.i.i = call i32 (%struct.FILE*, i8*, ...)* @fscanf( %struct.FILE* %tmp61, i8* getelementptr ([5 x i8]* @.str584, i32 0, i32 0), [1024 x i8]* null )		; <i32> [#uses=0]
	%tmp648.i.i = load i32* null, align 4		; <i32> [#uses=1]
	%tmp650.i468.i = icmp sgt i32 0, %tmp648.i.i		; <i1> [#uses=1]
	%tmp624.i469.i = call i32 (%struct.FILE*, i8*, ...)* @fscanf( %struct.FILE* %tmp61, i8* getelementptr ([5 x i8]* @.str584, i32 0, i32 0), [1024 x i8]* null )		; <i32> [#uses=0]
	%indvar.next167.i.i = add i32 %indvar166.i465.i, 1		; <i32> [#uses=1]
	br i1 %tmp650.i468.i, label %bb653.i.i.loopexit, label %bb620.i.i
bb653.i.i.loopexit:		; preds = %bb620.i.i
	%tmp642.i466.i = add i32 0, 1		; <i32> [#uses=1]
	br label %bb894.i.i
bb661.i.i:		; preds = %cond_next602.i.i
	ret void
bb894.loopexit.i.i:		; preds = %bb.i.i347.i
	br label %bb894.i.i
bb894.i.i:		; preds = %bb894.loopexit.i.i, %bb653.i.i.loopexit, %bb581.i.i, %bb177.i393.i
	%pinctr.0.i.i = phi i32 [ 0, %bb894.loopexit.i.i ], [ %tmp642.i466.i, %bb653.i.i.loopexit ], [ %pinctr.0.i.i, %bb177.i393.i ], [ %pinctr.0.i.i, %bb581.i.i ]		; <i32> [#uses=2]
	%soft.0.i.i = phi i32 [ undef, %bb894.loopexit.i.i ], [ %soft.0.i.i, %bb653.i.i.loopexit ], [ 0, %bb177.i393.i ], [ 1, %bb581.i.i ]		; <i32> [#uses=1]
	%cell.0.i.i = phi i32 [ 0, %bb894.loopexit.i.i ], [ %cell.0.i.i, %bb653.i.i.loopexit ], [ 0, %bb177.i393.i ], [ %tmp394.i418.i, %bb581.i.i ]		; <i32> [#uses=2]
	%ptr.0.i.i = phi %struct.cellbox* [ undef, %bb894.loopexit.i.i ], [ %ptr.0.i.i, %bb653.i.i.loopexit ], [ null, %bb177.i393.i ], [ %tmp397.i420.i, %bb581.i.i ]		; <%struct.cellbox*> [#uses=1]
	br i1 false, label %bb.i350.i, label %bb902.i502.i
bb902.i502.i:		; preds = %bb894.i.i
	ret void
cond_false674:		; preds = %entry
	ret void
}

	%struct.III_psy_xmin = type { [22 x double], [13 x [3 x double]] }
	%struct.III_scalefac_t = type { [22 x i32], [13 x [3 x i32]] }
	%struct.gr_info = type { i32, i32, i32, i32, i32, i32, i32, i32, [3 x i32], [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32*, [4 x i32] }
	%struct.lame_global_flags = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8*, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, float, float, float, float, i32, i32, i32, i32, i32, i32, i32, i32 }
@scalefac_band.1 = external global [14 x i32]		; <[14 x i32]*> [#uses=2]

declare fastcc i32 @init_outer_loop(%struct.lame_global_flags*, double*, %struct.gr_info*)

define fastcc void @outer_loop(%struct.lame_global_flags* %gfp, double* %xr, i32 %targ_bits, double* %best_noise, %struct.III_psy_xmin* %l3_xmin, i32* %l3_enc, %struct.III_scalefac_t* %scalefac, %struct.gr_info* %cod_info, i32 %ch) {
entry:
	%cod_info.182 = getelementptr %struct.gr_info* %cod_info, i32 0, i32 1		; <i32*> [#uses=1]
	br label %bb
bb:		; preds = %bb226, %entry
	%save_cod_info.1.1 = phi i32 [ undef, %entry ], [ %save_cod_info.1.1, %bb226 ]		; <i32> [#uses=2]
	br i1 false, label %cond_next, label %cond_true
cond_true:		; preds = %bb
	ret void
cond_next:		; preds = %bb
	br i1 false, label %cond_next144, label %cond_false
cond_false:		; preds = %cond_next
	ret void
cond_next144:		; preds = %cond_next
	br i1 false, label %cond_next205, label %cond_true163
cond_true163:		; preds = %cond_next144
	br i1 false, label %bb34.i, label %bb.i53
bb.i53:		; preds = %cond_true163
	ret void
bb34.i:		; preds = %cond_true163
	%tmp37.i55 = load i32* null, align 4		; <i32> [#uses=1]
	br i1 false, label %bb65.preheader.i, label %bb78.i
bb65.preheader.i:		; preds = %bb34.i
	br label %bb65.outer.us.i
bb65.outer.us.i:		; preds = %bb65.outer.us.i, %bb65.preheader.i
	br i1 false, label %bb78.i, label %bb65.outer.us.i
bb78.i:		; preds = %bb65.outer.us.i, %bb34.i
	br i1 false, label %bb151.i.preheader, label %bb90.i
bb90.i:		; preds = %bb78.i
	ret void
bb151.i.preheader:		; preds = %bb78.i
	br label %bb151.i
bb151.i:		; preds = %bb226.backedge.i, %bb151.i.preheader
	%i.154.i = phi i32 [ %tmp15747.i, %bb226.backedge.i ], [ 0, %bb151.i.preheader ]		; <i32> [#uses=2]
	%tmp15747.i = add i32 %i.154.i, 1		; <i32> [#uses=3]
	br i1 false, label %bb155.i, label %bb226.backedge.i
bb226.backedge.i:		; preds = %cond_next215.i, %bb151.i
	%tmp228.i71 = icmp slt i32 %tmp15747.i, 3		; <i1> [#uses=1]
	br i1 %tmp228.i71, label %bb151.i, label %amp_scalefac_bands.exit
bb155.i:		; preds = %cond_next215.i, %bb151.i
	%indvar90.i = phi i32 [ %indvar.next91.i, %cond_next215.i ], [ 0, %bb151.i ]		; <i32> [#uses=2]
	%sfb.3.reg2mem.0.i = add i32 %indvar90.i, %tmp37.i55		; <i32> [#uses=4]
	%tmp161.i = getelementptr [4 x [21 x double]]* null, i32 0, i32 %tmp15747.i, i32 %sfb.3.reg2mem.0.i		; <double*> [#uses=1]
	%tmp162.i74 = load double* %tmp161.i, align 4		; <double> [#uses=0]
	br i1 false, label %cond_true167.i, label %cond_next215.i
cond_true167.i:		; preds = %bb155.i
	%tmp173.i = getelementptr %struct.III_scalefac_t* null, i32 0, i32 1, i32 %sfb.3.reg2mem.0.i, i32 %i.154.i		; <i32*> [#uses=1]
	store i32 0, i32* %tmp173.i, align 4
	%tmp182.1.i = getelementptr [14 x i32]* @scalefac_band.1, i32 0, i32 %sfb.3.reg2mem.0.i		; <i32*> [#uses=0]
	%tmp185.i78 = add i32 %sfb.3.reg2mem.0.i, 1		; <i32> [#uses=1]
	%tmp187.1.i = getelementptr [14 x i32]* @scalefac_band.1, i32 0, i32 %tmp185.i78		; <i32*> [#uses=1]
	%tmp188.i = load i32* %tmp187.1.i, align 4		; <i32> [#uses=1]
	%tmp21153.i = icmp slt i32 0, %tmp188.i		; <i1> [#uses=1]
	br i1 %tmp21153.i, label %bb190.preheader.i, label %cond_next215.i
bb190.preheader.i:		; preds = %cond_true167.i
	ret void
cond_next215.i:		; preds = %cond_true167.i, %bb155.i
	%indvar.next91.i = add i32 %indvar90.i, 1		; <i32> [#uses=2]
	%exitcond99.i87 = icmp eq i32 %indvar.next91.i, 0		; <i1> [#uses=1]
	br i1 %exitcond99.i87, label %bb226.backedge.i, label %bb155.i
amp_scalefac_bands.exit:		; preds = %bb226.backedge.i
	br i1 false, label %bb19.i, label %bb.i16
bb.i16:		; preds = %amp_scalefac_bands.exit
	ret void
bb19.i:		; preds = %amp_scalefac_bands.exit
	br i1 false, label %bb40.outer.i, label %cond_next205
bb40.outer.i:		; preds = %bb19.i
	ret void
cond_next205:		; preds = %bb19.i, %cond_next144
	br i1 false, label %bb226, label %cond_true210
cond_true210:		; preds = %cond_next205
	br i1 false, label %bb226, label %cond_true217
cond_true217:		; preds = %cond_true210
	%tmp221 = call fastcc i32 @init_outer_loop( %struct.lame_global_flags* %gfp, double* %xr, %struct.gr_info* %cod_info )		; <i32> [#uses=0]
	ret void
bb226:		; preds = %cond_true210, %cond_next205
	br i1 false, label %bb231, label %bb
bb231:		; preds = %bb226
	store i32 %save_cod_info.1.1, i32* %cod_info.182
	ret void
}

	%struct.III_psy_xmin = type { [22 x double], [13 x [3 x double]] }
	%struct.III_scalefac_t = type { [22 x i32], [13 x [3 x i32]] }
	%struct.gr_info = type { i32, i32, i32, i32, i32, i32, i32, i32, [3 x i32], [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32*, [4 x i32] }
	%struct.lame_global_flags = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8*, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, float, float, float, float, i32, i32, i32, i32, i32, i32, i32, i32 }

define fastcc void @outer_loop2(%struct.lame_global_flags* %gfp, double* %xr, i32 %targ_bits, double* %best_noise, %struct.III_psy_xmin* %l3_xmin, i32* %l3_enc, %struct.III_scalefac_t* %scalefac, %struct.gr_info* %cod_info, i32 %ch) {
entry:
	%cod_info.20128.1 = getelementptr %struct.gr_info* %cod_info, i32 0, i32 20, i32 1		; <i32*> [#uses=1]
	%cod_info.20128.2 = getelementptr %struct.gr_info* %cod_info, i32 0, i32 20, i32 2		; <i32*> [#uses=1]
	%cod_info.20128.3 = getelementptr %struct.gr_info* %cod_info, i32 0, i32 20, i32 3		; <i32*> [#uses=1]
	br label %bb
bb:		; preds = %bb226, %entry
	%save_cod_info.19.1 = phi i32* [ undef, %entry ], [ %save_cod_info.19.0, %bb226 ]		; <i32*> [#uses=1]
	%save_cod_info.0.1 = phi i32 [ undef, %entry ], [ %save_cod_info.0.0, %bb226 ]		; <i32> [#uses=1]
	br i1 false, label %cond_next144, label %cond_false
cond_false:		; preds = %bb
	br i1 false, label %cond_true56, label %cond_false78
cond_true56:		; preds = %cond_false
	br i1 false, label %inner_loop.exit, label %cond_next85
inner_loop.exit:		; preds = %cond_true56
	br i1 false, label %cond_next104, label %cond_false96
cond_false78:		; preds = %cond_false
	ret void
cond_next85:		; preds = %cond_true56
	ret void
cond_false96:		; preds = %inner_loop.exit
	ret void
cond_next104:		; preds = %inner_loop.exit
	br i1 false, label %cond_next144, label %cond_false110
cond_false110:		; preds = %cond_next104
	ret void
cond_next144:		; preds = %cond_next104, %bb
	%save_cod_info.19.0 = phi i32* [ %save_cod_info.19.1, %bb ], [ null, %cond_next104 ]		; <i32*> [#uses=1]
	%save_cod_info.4.0 = phi i32 [ 0, %bb ], [ 0, %cond_next104 ]		; <i32> [#uses=1]
	%save_cod_info.3.0 = phi i32 [ 0, %bb ], [ 0, %cond_next104 ]		; <i32> [#uses=1]
	%save_cod_info.2.0 = phi i32 [ 0, %bb ], [ 0, %cond_next104 ]		; <i32> [#uses=1]
	%save_cod_info.1.0 = phi i32 [ 0, %bb ], [ 0, %cond_next104 ]		; <i32> [#uses=1]
	%save_cod_info.0.0 = phi i32 [ %save_cod_info.0.1, %bb ], [ 0, %cond_next104 ]		; <i32> [#uses=1]
	%over.1 = phi i32 [ 0, %bb ], [ 0, %cond_next104 ]		; <i32> [#uses=1]
	%best_over.0 = phi i32 [ 0, %bb ], [ 0, %cond_next104 ]		; <i32> [#uses=1]
	%notdone.0 = phi i32 [ 0, %bb ], [ 0, %cond_next104 ]		; <i32> [#uses=1]
	%tmp147 = load i32* null, align 4		; <i32> [#uses=1]
	%tmp148 = icmp eq i32 %tmp147, 0		; <i1> [#uses=1]
	%tmp153 = icmp eq i32 %over.1, 0		; <i1> [#uses=1]
	%bothcond = and i1 %tmp148, %tmp153		; <i1> [#uses=1]
	%notdone.2 = select i1 %bothcond, i32 0, i32 %notdone.0		; <i32> [#uses=1]
	br i1 false, label %cond_next205, label %cond_true163
cond_true163:		; preds = %cond_next144
	ret void
cond_next205:		; preds = %cond_next144
	br i1 false, label %bb226, label %cond_true210
cond_true210:		; preds = %cond_next205
	ret void
bb226:		; preds = %cond_next205
	%tmp228 = icmp eq i32 %notdone.2, 0		; <i1> [#uses=1]
	br i1 %tmp228, label %bb231, label %bb
bb231:		; preds = %bb226
	store i32 %save_cod_info.1.0, i32* null
	store i32 %save_cod_info.2.0, i32* null
	store i32 %save_cod_info.3.0, i32* null
	store i32 %save_cod_info.4.0, i32* null
	store i32 0, i32* %cod_info.20128.1
	store i32 0, i32* %cod_info.20128.2
	store i32 0, i32* %cod_info.20128.3
	%tmp244245 = sitofp i32 %best_over.0 to double		; <double> [#uses=1]
	store double %tmp244245, double* %best_noise, align 4
	ret void
}
