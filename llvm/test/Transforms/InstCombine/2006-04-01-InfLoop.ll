; RUN: llvm-as < %s | opt -instcombine -disable-output
; END.
	%struct.DecRefPicMarking_s = type { i32, i32, i32, i32, i32, %struct.DecRefPicMarking_s* }
	%struct.datapartition = type { %typedef.Bitstream*, %typedef.DecodingEnvironment, i32 (%struct.syntaxelement*, %struct.img_par*, %struct.inp_par*, %struct.datapartition*)* }
	%struct.img_par = type { i32, i32, i32, i32, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [16 x [16 x i16]], [6 x [32 x i32]], [16 x [16 x i32]], [4 x [12 x [4 x [4 x i32]]]], [16 x i32], i32**, i32*, i32***, i32**, i32, i32, i32, i32, %typedef.Slice*, %struct.macroblock*, i32, i32, i32, i32, i32, i32, i32**, %struct.DecRefPicMarking_s*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32***, i32***, i32****, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.timeb, %struct.timeb, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.inp_par = type { [100 x i8], [100 x i8], [100 x i8], i32, i32, i32, i32, i32, i32, i32 }
	%struct.macroblock = type { i32, i32, i32, %struct.macroblock*, %struct.macroblock*, i32, [2 x [4 x [4 x [2 x i32]]]], i32, i64, i64, i32, i32, [4 x i32], [4 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.pix_pos = type { i32, i32, i32, i32, i32, i32 }
	%struct.storable_picture = type { i32, i32, i32, i32, i32, [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], i32, i32, i32, i32, i32, i32, i32, i32, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i16**, i16***, i8*, i16**, i8***, i64***, i64***, i16****, i8**, i8**, %struct.storable_picture*, %struct.storable_picture*, %struct.storable_picture*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], i32, %struct.DecRefPicMarking_s*, i32 }
	%struct.syntaxelement = type { i32, i32, i32, i32, i32, i32, i32, i32, void (i32, i32, i32*, i32*)*, void (%struct.syntaxelement*, %struct.inp_par*, %struct.img_par*, %typedef.DecodingEnvironment*)* }
	%struct.timeb = type { i32, i16, i16, i16 }
	%typedef.BiContextType = type { i16, i8 }
	%typedef.Bitstream = type { i32, i32, i32, i32, i8*, i32 }
	%typedef.DecodingEnvironment = type { i32, i32, i32, i32, i32, i8*, i32* }
	%typedef.MotionInfoContexts = type { [4 x [11 x %typedef.BiContextType]], [2 x [9 x %typedef.BiContextType]], [2 x [10 x %typedef.BiContextType]], [2 x [6 x %typedef.BiContextType]], [4 x %typedef.BiContextType], [4 x %typedef.BiContextType], [3 x %typedef.BiContextType] }
	%typedef.Slice = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.datapartition*, %typedef.MotionInfoContexts*, %typedef.TextureInfoContexts*, i32, i32*, i32*, i32*, i32, i32*, i32*, i32*, i32 (%struct.img_par*, %struct.inp_par*)*, i32, i32, i32, i32 }
	%typedef.TextureInfoContexts = type { [2 x %typedef.BiContextType], [4 x %typedef.BiContextType], [3 x [4 x %typedef.BiContextType]], [10 x [4 x %typedef.BiContextType]], [10 x [15 x %typedef.BiContextType]], [10 x [15 x %typedef.BiContextType]], [10 x [5 x %typedef.BiContextType]], [10 x [5 x %typedef.BiContextType]], [10 x [15 x %typedef.BiContextType]], [10 x [15 x %typedef.BiContextType]] }
@dec_picture = external global %struct.storable_picture*		; <%struct.storable_picture**> [#uses=1]
@last_dquant = external global i32		; <i32*> [#uses=1]

define void @readCBP_CABAC(%struct.syntaxelement* %se, %struct.inp_par* %inp, %struct.img_par* %img.1, %typedef.DecodingEnvironment* %dep_dp) {
entry:
	%block_a = alloca %struct.pix_pos		; <%struct.pix_pos*> [#uses=5]
	%tmp.1 = getelementptr %struct.img_par* %img.1, i32 0, i32 37		; <%typedef.Slice**> [#uses=1]
	%tmp.2 = load %typedef.Slice** %tmp.1		; <%typedef.Slice*> [#uses=1]
	%tmp.3 = getelementptr %typedef.Slice* %tmp.2, i32 0, i32 11		; <%typedef.TextureInfoContexts**> [#uses=1]
	%tmp.4 = load %typedef.TextureInfoContexts** %tmp.3		; <%typedef.TextureInfoContexts*> [#uses=3]
	%tmp.6 = getelementptr %struct.img_par* %img.1, i32 0, i32 38		; <%struct.macroblock**> [#uses=1]
	%tmp.7 = load %struct.macroblock** %tmp.6		; <%struct.macroblock*> [#uses=1]
	%tmp.9 = getelementptr %struct.img_par* %img.1, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp.10 = load i32* %tmp.9		; <i32> [#uses=1]
	%tmp.11 = bitcast i32 %tmp.10 to i32		; <i32> [#uses=1]
	%tmp.12 = getelementptr %struct.macroblock* %tmp.7, i32 %tmp.11		; <%struct.macroblock*> [#uses=18]
	br label %loopentry.0

loopentry.0:		; preds = %loopexit.1, %entry
	%mask.1 = phi i32 [ undef, %entry ], [ %mask.0, %loopexit.1 ]		; <i32> [#uses=1]
	%cbp_bit.1 = phi i32 [ undef, %entry ], [ %cbp_bit.0, %loopexit.1 ]		; <i32> [#uses=1]
	%cbp.2 = phi i32 [ 0, %entry ], [ %cbp.1, %loopexit.1 ]		; <i32> [#uses=5]
	%curr_cbp_ctx.1 = phi i32 [ undef, %entry ], [ %curr_cbp_ctx.0, %loopexit.1 ]		; <i32> [#uses=1]
	%b.2 = phi i32 [ undef, %entry ], [ %b.1, %loopexit.1 ]		; <i32> [#uses=1]
	%a.2 = phi i32 [ undef, %entry ], [ %a.1, %loopexit.1 ]		; <i32> [#uses=1]
	%mb_y.0 = phi i32 [ 0, %entry ], [ %tmp.152, %loopexit.1 ]		; <i32> [#uses=7]
	%mb_x.0 = phi i32 [ undef, %entry ], [ %mb_x.1, %loopexit.1 ]		; <i32> [#uses=0]
	%tmp.14 = icmp sle i32 %mb_y.0, 3		; <i1> [#uses=2]
	%tmp.15 = zext i1 %tmp.14 to i32		; <i32> [#uses=0]
	br i1 %tmp.14, label %no_exit.0, label %loopexit.0

no_exit.0:		; preds = %loopentry.0
	br label %loopentry.1

loopentry.1:		; preds = %endif.7, %no_exit.0
	%mask.0 = phi i32 [ %mask.1, %no_exit.0 ], [ %tmp.131, %endif.7 ]		; <i32> [#uses=1]
	%cbp_bit.0 = phi i32 [ %cbp_bit.1, %no_exit.0 ], [ %tmp.142, %endif.7 ]		; <i32> [#uses=1]
	%cbp.1 = phi i32 [ %cbp.2, %no_exit.0 ], [ %cbp.0, %endif.7 ]		; <i32> [#uses=5]
	%curr_cbp_ctx.0 = phi i32 [ %curr_cbp_ctx.1, %no_exit.0 ], [ %tmp.125, %endif.7 ]		; <i32> [#uses=1]
	%b.1 = phi i32 [ %b.2, %no_exit.0 ], [ %b.0, %endif.7 ]		; <i32> [#uses=1]
	%a.1 = phi i32 [ %a.2, %no_exit.0 ], [ %a.0, %endif.7 ]		; <i32> [#uses=1]
	%mb_x.1 = phi i32 [ 0, %no_exit.0 ], [ %tmp.150, %endif.7 ]		; <i32> [#uses=9]
	%tmp.17 = icmp sle i32 %mb_x.1, 3		; <i1> [#uses=2]
	%tmp.18 = zext i1 %tmp.17 to i32		; <i32> [#uses=0]
	br i1 %tmp.17, label %no_exit.1, label %loopexit.1

no_exit.1:		; preds = %loopentry.1
	%tmp.20 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 12		; <[4 x i32]*> [#uses=1]
	%tmp.22 = sdiv i32 %mb_x.1, 2		; <i32> [#uses=1]
	%tmp.24 = add i32 %tmp.22, %mb_y.0		; <i32> [#uses=1]
	%tmp.25 = getelementptr [4 x i32]* %tmp.20, i32 0, i32 %tmp.24		; <i32*> [#uses=1]
	%tmp.26 = load i32* %tmp.25		; <i32> [#uses=1]
	%tmp.27 = icmp eq i32 %tmp.26, 11		; <i1> [#uses=2]
	%tmp.28 = zext i1 %tmp.27 to i32		; <i32> [#uses=0]
	br i1 %tmp.27, label %then.0, label %else.0

then.0:		; preds = %no_exit.1
	br label %endif.0

else.0:		; preds = %no_exit.1
	br label %endif.0

endif.0:		; preds = %else.0, %then.0
	%tmp.30 = icmp eq i32 %mb_y.0, 0		; <i1> [#uses=2]
	%tmp.31 = zext i1 %tmp.30 to i32		; <i32> [#uses=0]
	br i1 %tmp.30, label %then.1, label %else.1

then.1:		; preds = %endif.0
	%tmp.33 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 3		; <%struct.macroblock**> [#uses=1]
	%tmp.34 = load %struct.macroblock** %tmp.33		; <%struct.macroblock*> [#uses=1]
	%tmp.35 = bitcast %struct.macroblock* %tmp.34 to i8*		; <i8*> [#uses=1]
	%tmp.36 = icmp eq i8* %tmp.35, null		; <i1> [#uses=2]
	%tmp.37 = zext i1 %tmp.36 to i32		; <i32> [#uses=0]
	br i1 %tmp.36, label %then.2, label %else.2

then.2:		; preds = %then.1
	br label %endif.1

else.2:		; preds = %then.1
	%tmp.39 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 3		; <%struct.macroblock**> [#uses=1]
	%tmp.40 = load %struct.macroblock** %tmp.39		; <%struct.macroblock*> [#uses=1]
	%tmp.41 = getelementptr %struct.macroblock* %tmp.40, i32 0, i32 5		; <i32*> [#uses=1]
	%tmp.42 = load i32* %tmp.41		; <i32> [#uses=1]
	%tmp.43 = icmp eq i32 %tmp.42, 14		; <i1> [#uses=2]
	%tmp.44 = zext i1 %tmp.43 to i32		; <i32> [#uses=0]
	br i1 %tmp.43, label %then.3, label %else.3

then.3:		; preds = %else.2
	br label %endif.1

else.3:		; preds = %else.2
	%tmp.46 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 3		; <%struct.macroblock**> [#uses=1]
	%tmp.47 = load %struct.macroblock** %tmp.46		; <%struct.macroblock*> [#uses=1]
	%tmp.48 = getelementptr %struct.macroblock* %tmp.47, i32 0, i32 7		; <i32*> [#uses=1]
	%tmp.49 = load i32* %tmp.48		; <i32> [#uses=1]
	%tmp.51 = sdiv i32 %mb_x.1, 2		; <i32> [#uses=1]
	%tmp.52 = add i32 %tmp.51, 2		; <i32> [#uses=1]
	%tmp.53 = trunc i32 %tmp.52 to i8		; <i8> [#uses=1]
	%shift.upgrd.1 = zext i8 %tmp.53 to i32		; <i32> [#uses=1]
	%tmp.54 = ashr i32 %tmp.49, %shift.upgrd.1		; <i32> [#uses=1]
	%tmp.55 = bitcast i32 %tmp.54 to i32		; <i32> [#uses=1]
	%tmp.57 = xor i32 %tmp.55, 1		; <i32> [#uses=1]
	%tmp.58 = bitcast i32 %tmp.57 to i32		; <i32> [#uses=1]
	%tmp.59 = and i32 %tmp.58, 1		; <i32> [#uses=1]
	br label %endif.1

else.1:		; preds = %endif.0
	%tmp.62 = sdiv i32 %mb_x.1, 2		; <i32> [#uses=1]
	%tmp.63 = trunc i32 %tmp.62 to i8		; <i8> [#uses=1]
	%shift.upgrd.2 = zext i8 %tmp.63 to i32		; <i32> [#uses=1]
	%tmp.64 = ashr i32 %cbp.1, %shift.upgrd.2		; <i32> [#uses=1]
	%tmp.65 = bitcast i32 %tmp.64 to i32		; <i32> [#uses=1]
	%tmp.67 = xor i32 %tmp.65, 1		; <i32> [#uses=1]
	%tmp.68 = bitcast i32 %tmp.67 to i32		; <i32> [#uses=1]
	%tmp.69 = and i32 %tmp.68, 1		; <i32> [#uses=1]
	br label %endif.1

endif.1:		; preds = %else.1, %else.3, %then.3, %then.2
	%b.0 = phi i32 [ 0, %then.2 ], [ 0, %then.3 ], [ %tmp.59, %else.3 ], [ %tmp.69, %else.1 ]		; <i32> [#uses=2]
	%tmp.71 = icmp eq i32 %mb_x.1, 0		; <i1> [#uses=2]
	%tmp.72 = zext i1 %tmp.71 to i32		; <i32> [#uses=0]
	br i1 %tmp.71, label %then.4, label %else.4

then.4:		; preds = %endif.1
	%tmp.74 = getelementptr %struct.img_par* %img.1, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp.75 = load i32* %tmp.74		; <i32> [#uses=1]
	%tmp.76 = bitcast i32 %tmp.75 to i32		; <i32> [#uses=1]
	call void @getLuma4x4Neighbour( i32 %tmp.76, i32 %mb_x.1, i32 %mb_y.0, i32 -1, i32 0, %struct.pix_pos* %block_a )
	%tmp.79 = getelementptr %struct.pix_pos* %block_a, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp.80 = load i32* %tmp.79		; <i32> [#uses=1]
	%tmp.81 = icmp ne i32 %tmp.80, 0		; <i1> [#uses=2]
	%tmp.82 = zext i1 %tmp.81 to i32		; <i32> [#uses=0]
	br i1 %tmp.81, label %then.5, label %else.5

then.5:		; preds = %then.4
	%tmp.84 = getelementptr %struct.img_par* %img.1, i32 0, i32 38		; <%struct.macroblock**> [#uses=1]
	%tmp.85 = load %struct.macroblock** %tmp.84		; <%struct.macroblock*> [#uses=1]
	%tmp.86 = getelementptr %struct.pix_pos* %block_a, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp.87 = load i32* %tmp.86		; <i32> [#uses=1]
	%tmp.88 = getelementptr %struct.macroblock* %tmp.85, i32 %tmp.87		; <%struct.macroblock*> [#uses=1]
	%tmp.89 = getelementptr %struct.macroblock* %tmp.88, i32 0, i32 5		; <i32*> [#uses=1]
	%tmp.90 = load i32* %tmp.89		; <i32> [#uses=1]
	%tmp.91 = icmp eq i32 %tmp.90, 14		; <i1> [#uses=2]
	%tmp.92 = zext i1 %tmp.91 to i32		; <i32> [#uses=0]
	br i1 %tmp.91, label %then.6, label %else.6

then.6:		; preds = %then.5
	br label %endif.4

else.6:		; preds = %then.5
	%tmp.94 = getelementptr %struct.img_par* %img.1, i32 0, i32 38		; <%struct.macroblock**> [#uses=1]
	%tmp.95 = load %struct.macroblock** %tmp.94		; <%struct.macroblock*> [#uses=1]
	%tmp.96 = getelementptr %struct.pix_pos* %block_a, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp.97 = load i32* %tmp.96		; <i32> [#uses=1]
	%tmp.98 = getelementptr %struct.macroblock* %tmp.95, i32 %tmp.97		; <%struct.macroblock*> [#uses=1]
	%tmp.99 = getelementptr %struct.macroblock* %tmp.98, i32 0, i32 7		; <i32*> [#uses=1]
	%tmp.100 = load i32* %tmp.99		; <i32> [#uses=1]
	%tmp.101 = getelementptr %struct.pix_pos* %block_a, i32 0, i32 3		; <i32*> [#uses=1]
	%tmp.102 = load i32* %tmp.101		; <i32> [#uses=1]
	%tmp.103 = sdiv i32 %tmp.102, 2		; <i32> [#uses=1]
	%tmp.104 = mul i32 %tmp.103, 2		; <i32> [#uses=1]
	%tmp.105 = add i32 %tmp.104, 1		; <i32> [#uses=1]
	%tmp.106 = trunc i32 %tmp.105 to i8		; <i8> [#uses=1]
	%shift.upgrd.3 = zext i8 %tmp.106 to i32		; <i32> [#uses=1]
	%tmp.107 = ashr i32 %tmp.100, %shift.upgrd.3		; <i32> [#uses=1]
	%tmp.108 = bitcast i32 %tmp.107 to i32		; <i32> [#uses=1]
	%tmp.110 = xor i32 %tmp.108, 1		; <i32> [#uses=1]
	%tmp.111 = bitcast i32 %tmp.110 to i32		; <i32> [#uses=1]
	%tmp.112 = and i32 %tmp.111, 1		; <i32> [#uses=1]
	br label %endif.4

else.5:		; preds = %then.4
	br label %endif.4

else.4:		; preds = %endif.1
	%tmp.115 = trunc i32 %mb_y.0 to i8		; <i8> [#uses=1]
	%shift.upgrd.4 = zext i8 %tmp.115 to i32		; <i32> [#uses=1]
	%tmp.116 = ashr i32 %cbp.1, %shift.upgrd.4		; <i32> [#uses=1]
	%tmp.117 = bitcast i32 %tmp.116 to i32		; <i32> [#uses=1]
	%tmp.119 = xor i32 %tmp.117, 1		; <i32> [#uses=1]
	%tmp.120 = bitcast i32 %tmp.119 to i32		; <i32> [#uses=1]
	%tmp.121 = and i32 %tmp.120, 1		; <i32> [#uses=1]
	br label %endif.4

endif.4:		; preds = %else.4, %else.5, %else.6, %then.6
	%a.0 = phi i32 [ 0, %then.6 ], [ %tmp.112, %else.6 ], [ 0, %else.5 ], [ %tmp.121, %else.4 ]		; <i32> [#uses=2]
	%tmp.123 = mul i32 %b.0, 2		; <i32> [#uses=1]
	%tmp.125 = add i32 %tmp.123, %a.0		; <i32> [#uses=2]
	%tmp.127 = sdiv i32 %mb_x.1, 2		; <i32> [#uses=1]
	%tmp.129 = add i32 %tmp.127, %mb_y.0		; <i32> [#uses=1]
	%tmp.130 = trunc i32 %tmp.129 to i8		; <i8> [#uses=1]
	%shift.upgrd.5 = zext i8 %tmp.130 to i32		; <i32> [#uses=1]
	%tmp.131 = shl i32 1, %shift.upgrd.5		; <i32> [#uses=2]
	%tmp.135 = getelementptr %typedef.TextureInfoContexts* %tmp.4, i32 0, i32 2		; <[3 x [4 x %typedef.BiContextType]]*> [#uses=1]
	%tmp.136 = getelementptr [3 x [4 x %typedef.BiContextType]]* %tmp.135, i32 0, i32 0		; <[4 x %typedef.BiContextType]*> [#uses=1]
	%tmp.137 = getelementptr [4 x %typedef.BiContextType]* %tmp.136, i32 0, i32 0		; <%typedef.BiContextType*> [#uses=1]
	%tmp.139 = bitcast i32 %tmp.125 to i32		; <i32> [#uses=1]
	%tmp.140 = bitcast i32 %tmp.139 to i32		; <i32> [#uses=1]
	%tmp.141 = getelementptr %typedef.BiContextType* %tmp.137, i32 %tmp.140		; <%typedef.BiContextType*> [#uses=1]
	%tmp.132 = call i32 @biari_decode_symbol( %typedef.DecodingEnvironment* %dep_dp, %typedef.BiContextType* %tmp.141 )		; <i32> [#uses=1]
	%tmp.142 = bitcast i32 %tmp.132 to i32		; <i32> [#uses=2]
	%tmp.144 = icmp ne i32 %tmp.142, 0		; <i1> [#uses=2]
	%tmp.145 = zext i1 %tmp.144 to i32		; <i32> [#uses=0]
	br i1 %tmp.144, label %then.7, label %endif.7

then.7:		; preds = %endif.4
	%tmp.148 = add i32 %cbp.1, %tmp.131		; <i32> [#uses=1]
	br label %endif.7

endif.7:		; preds = %then.7, %endif.4
	%cbp.0 = phi i32 [ %tmp.148, %then.7 ], [ %cbp.1, %endif.4 ]		; <i32> [#uses=1]
	%tmp.150 = add i32 %mb_x.1, 2		; <i32> [#uses=1]
	br label %loopentry.1

loopexit.1:		; preds = %loopentry.1
	%tmp.152 = add i32 %mb_y.0, 2		; <i32> [#uses=1]
	br label %loopentry.0

loopexit.0:		; preds = %loopentry.0
	%tmp.153 = load %struct.storable_picture** @dec_picture		; <%struct.storable_picture*> [#uses=1]
	%tmp.154 = getelementptr %struct.storable_picture* %tmp.153, i32 0, i32 45		; <i32*> [#uses=1]
	%tmp.155 = load i32* %tmp.154		; <i32> [#uses=1]
	%tmp.156 = icmp ne i32 %tmp.155, 0		; <i1> [#uses=2]
	%tmp.157 = zext i1 %tmp.156 to i32		; <i32> [#uses=0]
	br i1 %tmp.156, label %then.8, label %endif.8

then.8:		; preds = %loopexit.0
	%tmp.159 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 3		; <%struct.macroblock**> [#uses=1]
	%tmp.160 = load %struct.macroblock** %tmp.159		; <%struct.macroblock*> [#uses=1]
	%tmp.161 = bitcast %struct.macroblock* %tmp.160 to i8*		; <i8*> [#uses=1]
	%tmp.162 = icmp ne i8* %tmp.161, null		; <i1> [#uses=2]
	%tmp.163 = zext i1 %tmp.162 to i32		; <i32> [#uses=0]
	br i1 %tmp.162, label %then.9, label %endif.9

then.9:		; preds = %then.8
	%tmp.165 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 3		; <%struct.macroblock**> [#uses=1]
	%tmp.166 = load %struct.macroblock** %tmp.165		; <%struct.macroblock*> [#uses=1]
	%tmp.167 = getelementptr %struct.macroblock* %tmp.166, i32 0, i32 5		; <i32*> [#uses=1]
	%tmp.168 = load i32* %tmp.167		; <i32> [#uses=1]
	%tmp.169 = icmp eq i32 %tmp.168, 14		; <i1> [#uses=2]
	%tmp.170 = zext i1 %tmp.169 to i32		; <i32> [#uses=0]
	br i1 %tmp.169, label %then.10, label %else.7

then.10:		; preds = %then.9
	br label %endif.9

else.7:		; preds = %then.9
	%tmp.172 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 3		; <%struct.macroblock**> [#uses=1]
	%tmp.173 = load %struct.macroblock** %tmp.172		; <%struct.macroblock*> [#uses=1]
	%tmp.174 = getelementptr %struct.macroblock* %tmp.173, i32 0, i32 7		; <i32*> [#uses=1]
	%tmp.175 = load i32* %tmp.174		; <i32> [#uses=1]
	%tmp.176 = icmp sgt i32 %tmp.175, 15		; <i1> [#uses=1]
	%tmp.177 = zext i1 %tmp.176 to i32		; <i32> [#uses=1]
	br label %endif.9

endif.9:		; preds = %else.7, %then.10, %then.8
	%b.4 = phi i32 [ 1, %then.10 ], [ %tmp.177, %else.7 ], [ 0, %then.8 ]		; <i32> [#uses=1]
	%tmp.179 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 4		; <%struct.macroblock**> [#uses=1]
	%tmp.180 = load %struct.macroblock** %tmp.179		; <%struct.macroblock*> [#uses=1]
	%tmp.181 = bitcast %struct.macroblock* %tmp.180 to i8*		; <i8*> [#uses=1]
	%tmp.182 = icmp ne i8* %tmp.181, null		; <i1> [#uses=2]
	%tmp.183 = zext i1 %tmp.182 to i32		; <i32> [#uses=0]
	br i1 %tmp.182, label %then.11, label %endif.11

then.11:		; preds = %endif.9
	%tmp.185 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 4		; <%struct.macroblock**> [#uses=1]
	%tmp.186 = load %struct.macroblock** %tmp.185		; <%struct.macroblock*> [#uses=1]
	%tmp.187 = getelementptr %struct.macroblock* %tmp.186, i32 0, i32 5		; <i32*> [#uses=1]
	%tmp.188 = load i32* %tmp.187		; <i32> [#uses=1]
	%tmp.189 = icmp eq i32 %tmp.188, 14		; <i1> [#uses=2]
	%tmp.190 = zext i1 %tmp.189 to i32		; <i32> [#uses=0]
	br i1 %tmp.189, label %then.12, label %else.8

then.12:		; preds = %then.11
	br label %endif.11

else.8:		; preds = %then.11
	%tmp.192 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 4		; <%struct.macroblock**> [#uses=1]
	%tmp.193 = load %struct.macroblock** %tmp.192		; <%struct.macroblock*> [#uses=1]
	%tmp.194 = getelementptr %struct.macroblock* %tmp.193, i32 0, i32 7		; <i32*> [#uses=1]
	%tmp.195 = load i32* %tmp.194		; <i32> [#uses=1]
	%tmp.196 = icmp sgt i32 %tmp.195, 15		; <i1> [#uses=1]
	%tmp.197 = zext i1 %tmp.196 to i32		; <i32> [#uses=1]
	br label %endif.11

endif.11:		; preds = %else.8, %then.12, %endif.9
	%a.4 = phi i32 [ 1, %then.12 ], [ %tmp.197, %else.8 ], [ 0, %endif.9 ]		; <i32> [#uses=1]
	%tmp.199 = mul i32 %b.4, 2		; <i32> [#uses=1]
	%tmp.201 = add i32 %tmp.199, %a.4		; <i32> [#uses=1]
	%tmp.205 = getelementptr %typedef.TextureInfoContexts* %tmp.4, i32 0, i32 2		; <[3 x [4 x %typedef.BiContextType]]*> [#uses=1]
	%tmp.206 = getelementptr [3 x [4 x %typedef.BiContextType]]* %tmp.205, i32 0, i32 1		; <[4 x %typedef.BiContextType]*> [#uses=1]
	%tmp.207 = getelementptr [4 x %typedef.BiContextType]* %tmp.206, i32 0, i32 0		; <%typedef.BiContextType*> [#uses=1]
	%tmp.209 = bitcast i32 %tmp.201 to i32		; <i32> [#uses=1]
	%tmp.210 = bitcast i32 %tmp.209 to i32		; <i32> [#uses=1]
	%tmp.211 = getelementptr %typedef.BiContextType* %tmp.207, i32 %tmp.210		; <%typedef.BiContextType*> [#uses=1]
	%tmp.202 = call i32 @biari_decode_symbol( %typedef.DecodingEnvironment* %dep_dp, %typedef.BiContextType* %tmp.211 )		; <i32> [#uses=1]
	%tmp.212 = bitcast i32 %tmp.202 to i32		; <i32> [#uses=1]
	%tmp.214 = icmp ne i32 %tmp.212, 0		; <i1> [#uses=2]
	%tmp.215 = zext i1 %tmp.214 to i32		; <i32> [#uses=0]
	br i1 %tmp.214, label %then.13, label %endif.8

then.13:		; preds = %endif.11
	%tmp.217 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 3		; <%struct.macroblock**> [#uses=1]
	%tmp.218 = load %struct.macroblock** %tmp.217		; <%struct.macroblock*> [#uses=1]
	%tmp.219 = bitcast %struct.macroblock* %tmp.218 to i8*		; <i8*> [#uses=1]
	%tmp.220 = icmp ne i8* %tmp.219, null		; <i1> [#uses=2]
	%tmp.221 = zext i1 %tmp.220 to i32		; <i32> [#uses=0]
	br i1 %tmp.220, label %then.14, label %endif.14

then.14:		; preds = %then.13
	%tmp.223 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 3		; <%struct.macroblock**> [#uses=1]
	%tmp.224 = load %struct.macroblock** %tmp.223		; <%struct.macroblock*> [#uses=1]
	%tmp.225 = getelementptr %struct.macroblock* %tmp.224, i32 0, i32 5		; <i32*> [#uses=1]
	%tmp.226 = load i32* %tmp.225		; <i32> [#uses=1]
	%tmp.227 = icmp eq i32 %tmp.226, 14		; <i1> [#uses=2]
	%tmp.228 = zext i1 %tmp.227 to i32		; <i32> [#uses=0]
	br i1 %tmp.227, label %then.15, label %else.9

then.15:		; preds = %then.14
	br label %endif.14

else.9:		; preds = %then.14
	%tmp.230 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 3		; <%struct.macroblock**> [#uses=1]
	%tmp.231 = load %struct.macroblock** %tmp.230		; <%struct.macroblock*> [#uses=1]
	%tmp.232 = getelementptr %struct.macroblock* %tmp.231, i32 0, i32 7		; <i32*> [#uses=1]
	%tmp.233 = load i32* %tmp.232		; <i32> [#uses=1]
	%tmp.234 = icmp sgt i32 %tmp.233, 15		; <i1> [#uses=2]
	%tmp.235 = zext i1 %tmp.234 to i32		; <i32> [#uses=0]
	br i1 %tmp.234, label %then.16, label %endif.14

then.16:		; preds = %else.9
	%tmp.237 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 3		; <%struct.macroblock**> [#uses=1]
	%tmp.238 = load %struct.macroblock** %tmp.237		; <%struct.macroblock*> [#uses=1]
	%tmp.239 = getelementptr %struct.macroblock* %tmp.238, i32 0, i32 7		; <i32*> [#uses=1]
	%tmp.240 = load i32* %tmp.239		; <i32> [#uses=1]
	%tmp.242 = ashr i32 %tmp.240, 4		; <i32> [#uses=1]
	%tmp.243 = icmp eq i32 %tmp.242, 2		; <i1> [#uses=1]
	%tmp.244 = zext i1 %tmp.243 to i32		; <i32> [#uses=1]
	br label %endif.14

endif.14:		; preds = %then.16, %else.9, %then.15, %then.13
	%b.5 = phi i32 [ 1, %then.15 ], [ %tmp.244, %then.16 ], [ 0, %else.9 ], [ 0, %then.13 ]		; <i32> [#uses=1]
	%tmp.246 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 4		; <%struct.macroblock**> [#uses=1]
	%tmp.247 = load %struct.macroblock** %tmp.246		; <%struct.macroblock*> [#uses=1]
	%tmp.248 = bitcast %struct.macroblock* %tmp.247 to i8*		; <i8*> [#uses=1]
	%tmp.249 = icmp ne i8* %tmp.248, null		; <i1> [#uses=2]
	%tmp.250 = zext i1 %tmp.249 to i32		; <i32> [#uses=0]
	br i1 %tmp.249, label %then.17, label %endif.17

then.17:		; preds = %endif.14
	%tmp.252 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 4		; <%struct.macroblock**> [#uses=1]
	%tmp.253 = load %struct.macroblock** %tmp.252		; <%struct.macroblock*> [#uses=1]
	%tmp.254 = getelementptr %struct.macroblock* %tmp.253, i32 0, i32 5		; <i32*> [#uses=1]
	%tmp.255 = load i32* %tmp.254		; <i32> [#uses=1]
	%tmp.256 = icmp eq i32 %tmp.255, 14		; <i1> [#uses=2]
	%tmp.257 = zext i1 %tmp.256 to i32		; <i32> [#uses=0]
	br i1 %tmp.256, label %then.18, label %else.10

then.18:		; preds = %then.17
	br label %endif.17

else.10:		; preds = %then.17
	%tmp.259 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 4		; <%struct.macroblock**> [#uses=1]
	%tmp.260 = load %struct.macroblock** %tmp.259		; <%struct.macroblock*> [#uses=1]
	%tmp.261 = getelementptr %struct.macroblock* %tmp.260, i32 0, i32 7		; <i32*> [#uses=1]
	%tmp.262 = load i32* %tmp.261		; <i32> [#uses=1]
	%tmp.263 = icmp sgt i32 %tmp.262, 15		; <i1> [#uses=2]
	%tmp.264 = zext i1 %tmp.263 to i32		; <i32> [#uses=0]
	br i1 %tmp.263, label %then.19, label %endif.17

then.19:		; preds = %else.10
	%tmp.266 = getelementptr %struct.macroblock* %tmp.12, i32 0, i32 4		; <%struct.macroblock**> [#uses=1]
	%tmp.267 = load %struct.macroblock** %tmp.266		; <%struct.macroblock*> [#uses=1]
	%tmp.268 = getelementptr %struct.macroblock* %tmp.267, i32 0, i32 7		; <i32*> [#uses=1]
	%tmp.269 = load i32* %tmp.268		; <i32> [#uses=1]
	%tmp.271 = ashr i32 %tmp.269, 4		; <i32> [#uses=1]
	%tmp.272 = icmp eq i32 %tmp.271, 2		; <i1> [#uses=1]
	%tmp.273 = zext i1 %tmp.272 to i32		; <i32> [#uses=1]
	br label %endif.17

endif.17:		; preds = %then.19, %else.10, %then.18, %endif.14
	%a.5 = phi i32 [ 1, %then.18 ], [ %tmp.273, %then.19 ], [ 0, %else.10 ], [ 0, %endif.14 ]		; <i32> [#uses=1]
	%tmp.275 = mul i32 %b.5, 2		; <i32> [#uses=1]
	%tmp.277 = add i32 %tmp.275, %a.5		; <i32> [#uses=1]
	%tmp.281 = getelementptr %typedef.TextureInfoContexts* %tmp.4, i32 0, i32 2		; <[3 x [4 x %typedef.BiContextType]]*> [#uses=1]
	%tmp.282 = getelementptr [3 x [4 x %typedef.BiContextType]]* %tmp.281, i32 0, i32 2		; <[4 x %typedef.BiContextType]*> [#uses=1]
	%tmp.283 = getelementptr [4 x %typedef.BiContextType]* %tmp.282, i32 0, i32 0		; <%typedef.BiContextType*> [#uses=1]
	%tmp.285 = bitcast i32 %tmp.277 to i32		; <i32> [#uses=1]
	%tmp.286 = bitcast i32 %tmp.285 to i32		; <i32> [#uses=1]
	%tmp.287 = getelementptr %typedef.BiContextType* %tmp.283, i32 %tmp.286		; <%typedef.BiContextType*> [#uses=1]
	%tmp.278 = call i32 @biari_decode_symbol( %typedef.DecodingEnvironment* %dep_dp, %typedef.BiContextType* %tmp.287 )		; <i32> [#uses=1]
	%tmp.288 = bitcast i32 %tmp.278 to i32		; <i32> [#uses=1]
	%tmp.290 = icmp eq i32 %tmp.288, 1		; <i1> [#uses=2]
	%tmp.291 = zext i1 %tmp.290 to i32		; <i32> [#uses=0]
	br i1 %tmp.290, label %cond_true, label %cond_false

cond_true:		; preds = %endif.17
	%tmp.293 = add i32 %cbp.2, 32		; <i32> [#uses=1]
	br label %cond_continue

cond_false:		; preds = %endif.17
	%tmp.295 = add i32 %cbp.2, 16		; <i32> [#uses=1]
	br label %cond_continue

cond_continue:		; preds = %cond_false, %cond_true
	%mem_tmp.0 = phi i32 [ %tmp.293, %cond_true ], [ %tmp.295, %cond_false ]		; <i32> [#uses=1]
	br label %endif.8

endif.8:		; preds = %cond_continue, %endif.11, %loopexit.0
	%cbp.3 = phi i32 [ %mem_tmp.0, %cond_continue ], [ %cbp.2, %endif.11 ], [ %cbp.2, %loopexit.0 ]		; <i32> [#uses=2]
	%tmp.298 = getelementptr %struct.syntaxelement* %se, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %cbp.3, i32* %tmp.298
	%tmp.301 = icmp eq i32 %cbp.3, 0		; <i1> [#uses=2]
	%tmp.302 = zext i1 %tmp.301 to i32		; <i32> [#uses=0]
	br i1 %tmp.301, label %then.20, label %return

then.20:		; preds = %endif.8
	store i32 0, i32* @last_dquant
	ret void

return:		; preds = %endif.8
	ret void
}

declare i32 @biari_decode_symbol(%typedef.DecodingEnvironment*, %typedef.BiContextType*)

declare void @getLuma4x4Neighbour(i32, i32, i32, i32, i32, %struct.pix_pos*)
