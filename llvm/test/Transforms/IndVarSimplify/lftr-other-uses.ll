; RUN: opt < %s -indvars -disable-output

; Don't RAUW the loop's original comparison instruction if it has
; other uses which aren't dominated by the new comparison instruction.

	%struct.DecRefPicMarking_s = type { i32, i32, i32, i32, i32, %struct.DecRefPicMarking_s* }
	%struct.datapartition = type { %typedef.Bitstream*, %typedef.DecodingEnvironment, i32 (%struct.syntaxelement*, %struct.img_par*, %struct.inp_par*, %struct.datapartition*)* }
	%struct.img_par = type { i32, i32, i32, i32, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [16 x [16 x i16]], [6 x [32 x i32]], [16 x [16 x i32]], [4 x [12 x [4 x [4 x i32]]]], [16 x i32], i32**, i32*, i32***, i32**, i32, i32, i32, i32, %typedef.Slice*, %struct.macroblock*, i32, i32, i32, i32, i32, i32, i32**, %struct.DecRefPicMarking_s*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32***, i32***, i32****, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.timeb, %struct.timeb, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.inp_par = type { [100 x i8], [100 x i8], [100 x i8], i32, i32, i32, i32, i32, i32, i32 }
	%struct.macroblock = type { i32, i32, i32, %struct.macroblock*, %struct.macroblock*, i32, [2 x [4 x [4 x [2 x i32]]]], i32, i64, i64, i32, i32, [4 x i32], [4 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.syntaxelement = type { i32, i32, i32, i32, i32, i32, i32, i32, void (i32, i32, i32*, i32*)*, void (%struct.syntaxelement*, %struct.inp_par*, %struct.img_par*, %typedef.DecodingEnvironment*)* }
	%struct.timeb = type { i32, i16, i16, i16 }
	%typedef.BiContextType = type { i16, i8 }
	%typedef.Bitstream = type { i32, i32, i32, i32, i8*, i32 }
	%typedef.DecodingEnvironment = type { i32, i32, i32, i32, i32, i8*, i32* }
	%typedef.MotionInfoContexts = type { [4 x [11 x %typedef.BiContextType]], [2 x [9 x %typedef.BiContextType]], [2 x [10 x %typedef.BiContextType]], [2 x [6 x %typedef.BiContextType]], [4 x %typedef.BiContextType], [4 x %typedef.BiContextType], [3 x %typedef.BiContextType] }
	%typedef.Slice = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.datapartition*, %typedef.MotionInfoContexts*, %typedef.TextureInfoContexts*, i32, i32*, i32*, i32*, i32, i32*, i32*, i32*, i32 (%struct.img_par*, %struct.inp_par*)*, i32, i32, i32, i32 }
	%typedef.TextureInfoContexts = type { [2 x %typedef.BiContextType], [4 x %typedef.BiContextType], [3 x [4 x %typedef.BiContextType]], [10 x [4 x %typedef.BiContextType]], [10 x [15 x %typedef.BiContextType]], [10 x [15 x %typedef.BiContextType]], [10 x [5 x %typedef.BiContextType]], [10 x [5 x %typedef.BiContextType]], [10 x [15 x %typedef.BiContextType]], [10 x [15 x %typedef.BiContextType]] }

define void @readCBP_CABAC(%struct.syntaxelement* %se, %struct.inp_par* %inp, %struct.img_par* %img.1, %typedef.DecodingEnvironment* %dep_dp) {
entry:
	br label %loopentry.0

loopentry.0:		; preds = %loopentry.1, %entry
	%mb_y.0 = phi i32 [ 0, %entry ], [ %tmp.152, %loopentry.1 ]		; <i32> [#uses=2]
	%tmp.14 = icmp sle i32 %mb_y.0, 3		; <i1> [#uses=2]
	%tmp.15 = zext i1 %tmp.14 to i32		; <i32> [#uses=0]
	br i1 %tmp.14, label %loopentry.1, label %loopexit.0

loopentry.1:		; preds = %loopentry.0
	%tmp.152 = add i32 %mb_y.0, 2		; <i32> [#uses=1]
	br label %loopentry.0

loopexit.0:		; preds = %loopentry.0
	unreachable
}
