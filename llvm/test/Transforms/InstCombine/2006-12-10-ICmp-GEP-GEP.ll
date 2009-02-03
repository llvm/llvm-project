; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    grep -v {icmp ult i32}
; END.

target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.edgeBox = type { i16, i16, i16, i16, i16, i16 }
@qsz = external global i32		; <i32*> [#uses=12]
@thresh = external global i32		; <i32*> [#uses=2]
@mthresh = external global i32		; <i32*> [#uses=1]

define i32 @qsorte(i8* %base, i32 %n, i32 %size) {
entry:
	%tmp = icmp sgt i32 %n, 1		; <i1> [#uses=1]
	br i1 %tmp, label %cond_next, label %return

cond_next:		; preds = %entry
	store i32 %size, i32* @qsz
	%tmp3 = shl i32 %size, 2		; <i32> [#uses=1]
	store i32 %tmp3, i32* @thresh
	%tmp4 = load i32* @qsz		; <i32> [#uses=1]
	%tmp5 = mul i32 %tmp4, 6		; <i32> [#uses=1]
	store i32 %tmp5, i32* @mthresh
	%tmp6 = load i32* @qsz		; <i32> [#uses=1]
	%tmp8 = mul i32 %tmp6, %n		; <i32> [#uses=1]
	%tmp9 = getelementptr i8* %base, i32 %tmp8		; <i8*> [#uses=3]
	%tmp11 = icmp sgt i32 %n, 3		; <i1> [#uses=1]
	br i1 %tmp11, label %cond_true12, label %bb30

cond_true12:		; preds = %cond_next
	%tmp156 = call i32 @qste( i8* %base, i8* %tmp9 )		; <i32> [#uses=0]
	%tmp16 = load i32* @thresh		; <i32> [#uses=1]
	%tmp18 = getelementptr i8* %base, i32 %tmp16		; <i8*> [#uses=2]
	%tmp3117 = load i32* @qsz		; <i32> [#uses=1]
	%tmp3318 = getelementptr i8* %base, i32 %tmp3117		; <i8*> [#uses=2]
	%tmp3621 = icmp ult i8* %tmp3318, %tmp18		; <i1> [#uses=1]
	br i1 %tmp3621, label %bb, label %bb37

bb:		; preds = %bb30, %cond_true12
	%hi.0.0 = phi i8* [ %tmp18, %cond_true12 ], [ %hi.0, %bb30 ]		; <i8*> [#uses=4]
	%j.1.0 = phi i8* [ %base, %cond_true12 ], [ %j.1, %bb30 ]		; <i8*> [#uses=4]
	%tmp33.0 = phi i8* [ %tmp3318, %cond_true12 ], [ %tmp33, %bb30 ]		; <i8*> [#uses=6]
	%tmp3.upgrd.1 = bitcast i8* %j.1.0 to %struct.edgeBox*		; <%struct.edgeBox*> [#uses=1]
	%tmp4.upgrd.2 = bitcast i8* %tmp33.0 to %struct.edgeBox*		; <%struct.edgeBox*> [#uses=1]
	%tmp255 = call i32 @comparee( %struct.edgeBox* %tmp3.upgrd.1, %struct.edgeBox* %tmp4.upgrd.2 )		; <i32> [#uses=1]
	%tmp26 = icmp sgt i32 %tmp255, 0		; <i1> [#uses=1]
	br i1 %tmp26, label %cond_true27, label %bb30

cond_true27:		; preds = %bb
	br label %bb30

bb30:		; preds = %cond_true27, %bb, %cond_next
	%hi.0.3 = phi i8* [ %hi.0.0, %cond_true27 ], [ %hi.0.0, %bb ], [ undef, %cond_next ]		; <i8*> [#uses=0]
	%j.1.3 = phi i8* [ %j.1.0, %cond_true27 ], [ %j.1.0, %bb ], [ undef, %cond_next ]		; <i8*> [#uses=0]
	%tmp33.3 = phi i8* [ %tmp33.0, %cond_true27 ], [ %tmp33.0, %bb ], [ undef, %cond_next ]		; <i8*> [#uses=0]
	%hi.0 = phi i8* [ %tmp9, %cond_next ], [ %hi.0.0, %bb ], [ %hi.0.0, %cond_true27 ]		; <i8*> [#uses=2]
	%lo.1 = phi i8* [ %tmp33.0, %cond_true27 ], [ %tmp33.0, %bb ], [ %base, %cond_next ]		; <i8*> [#uses=1]
	%j.1 = phi i8* [ %tmp33.0, %cond_true27 ], [ %j.1.0, %bb ], [ %base, %cond_next ]		; <i8*> [#uses=2]
	%tmp31 = load i32* @qsz		; <i32> [#uses=1]
	%tmp33 = getelementptr i8* %lo.1, i32 %tmp31		; <i8*> [#uses=2]
	%tmp36 = icmp ult i8* %tmp33, %hi.0		; <i1> [#uses=1]
	br i1 %tmp36, label %bb, label %bb37

bb37:		; preds = %bb30, %cond_true12
	%j.1.1 = phi i8* [ %j.1, %bb30 ], [ %base, %cond_true12 ]		; <i8*> [#uses=4]
	%tmp40 = icmp eq i8* %j.1.1, %base		; <i1> [#uses=1]
	br i1 %tmp40, label %bb115, label %cond_true41

cond_true41:		; preds = %bb37
	%tmp43 = load i32* @qsz		; <i32> [#uses=1]
	%tmp45 = getelementptr i8* %base, i32 %tmp43		; <i8*> [#uses=2]
	%tmp6030 = icmp ult i8* %base, %tmp45		; <i1> [#uses=1]
	br i1 %tmp6030, label %bb46, label %bb115

bb46:		; preds = %bb46, %cond_true41
	%j.2.0 = phi i8* [ %j.1.1, %cond_true41 ], [ %tmp52, %bb46 ]		; <i8*> [#uses=3]
	%i.2.0 = phi i8* [ %base, %cond_true41 ], [ %tmp56, %bb46 ]		; <i8*> [#uses=3]
	%tmp.upgrd.3 = load i8* %j.2.0		; <i8> [#uses=2]
	%tmp49 = load i8* %i.2.0		; <i8> [#uses=1]
	store i8 %tmp49, i8* %j.2.0
	%tmp52 = getelementptr i8* %j.2.0, i32 1		; <i8*> [#uses=2]
	store i8 %tmp.upgrd.3, i8* %i.2.0
	%tmp56 = getelementptr i8* %i.2.0, i32 1		; <i8*> [#uses=3]
	%tmp60 = icmp ult i8* %tmp56, %tmp45		; <i1> [#uses=1]
	br i1 %tmp60, label %bb46, label %bb115

bb66:		; preds = %bb115, %bb66
	%hi.3 = phi i8* [ %tmp118, %bb115 ], [ %tmp70, %bb66 ]		; <i8*> [#uses=2]
	%tmp67 = load i32* @qsz		; <i32> [#uses=2]
	%tmp68 = sub i32 0, %tmp67		; <i32> [#uses=1]
	%tmp70 = getelementptr i8* %hi.3, i32 %tmp68		; <i8*> [#uses=2]
	%tmp.upgrd.4 = bitcast i8* %tmp70 to %struct.edgeBox*		; <%struct.edgeBox*> [#uses=1]
	%tmp1 = bitcast i8* %tmp118 to %struct.edgeBox*		; <%struct.edgeBox*> [#uses=1]
	%tmp732 = call i32 @comparee( %struct.edgeBox* %tmp.upgrd.4, %struct.edgeBox* %tmp1 )		; <i32> [#uses=1]
	%tmp74 = icmp sgt i32 %tmp732, 0		; <i1> [#uses=1]
	br i1 %tmp74, label %bb66, label %bb75

bb75:		; preds = %bb66
	%tmp76 = load i32* @qsz		; <i32> [#uses=1]
	%tmp70.sum = sub i32 %tmp76, %tmp67		; <i32> [#uses=1]
	%tmp78 = getelementptr i8* %hi.3, i32 %tmp70.sum		; <i8*> [#uses=3]
	%tmp81 = icmp eq i8* %tmp78, %tmp118		; <i1> [#uses=1]
	br i1 %tmp81, label %bb115, label %cond_true82

cond_true82:		; preds = %bb75
	%tmp83 = load i32* @qsz		; <i32> [#uses=1]
	%tmp118.sum = add i32 %tmp116, %tmp83		; <i32> [#uses=1]
	%tmp85 = getelementptr i8* %min.1, i32 %tmp118.sum		; <i8*> [#uses=1]
	%tmp10937 = getelementptr i8* %tmp85, i32 -1		; <i8*> [#uses=3]
	%tmp11239 = icmp ult i8* %tmp10937, %tmp118		; <i1> [#uses=1]
	br i1 %tmp11239, label %bb115, label %bb86

bb86:		; preds = %bb104, %cond_true82
	%tmp109.0 = phi i8* [ %tmp10937, %cond_true82 ], [ %tmp109, %bb104 ]		; <i8*> [#uses=5]
	%i.5.2 = phi i8* [ %i.5.3, %cond_true82 ], [ %i.5.1, %bb104 ]		; <i8*> [#uses=0]
	%tmp100.2 = phi i8* [ %tmp100.3, %cond_true82 ], [ %tmp100.1, %bb104 ]		; <i8*> [#uses=0]
	%tmp88 = load i8* %tmp109.0		; <i8> [#uses=2]
	%tmp9746 = load i32* @qsz		; <i32> [#uses=1]
	%tmp9847 = sub i32 0, %tmp9746		; <i32> [#uses=1]
	%tmp10048 = getelementptr i8* %tmp109.0, i32 %tmp9847		; <i8*> [#uses=3]
	%tmp10350 = icmp ult i8* %tmp10048, %tmp78		; <i1> [#uses=1]
	br i1 %tmp10350, label %bb104, label %bb91

bb91:		; preds = %bb91, %bb86
	%i.5.0 = phi i8* [ %tmp109.0, %bb86 ], [ %tmp100.0, %bb91 ]		; <i8*> [#uses=1]
	%tmp100.0 = phi i8* [ %tmp10048, %bb86 ], [ %tmp100, %bb91 ]		; <i8*> [#uses=4]
	%tmp93 = load i8* %tmp100.0		; <i8> [#uses=1]
	store i8 %tmp93, i8* %i.5.0
	%tmp97 = load i32* @qsz		; <i32> [#uses=1]
	%tmp98 = sub i32 0, %tmp97		; <i32> [#uses=1]
	%tmp100 = getelementptr i8* %tmp100.0, i32 %tmp98		; <i8*> [#uses=3]
	%tmp103 = icmp ult i8* %tmp100, %tmp78		; <i1> [#uses=1]
	br i1 %tmp103, label %bb104, label %bb91

bb104:		; preds = %bb91, %bb86
	%i.5.1 = phi i8* [ %tmp109.0, %bb86 ], [ %tmp100.0, %bb91 ]		; <i8*> [#uses=4]
	%tmp100.1 = phi i8* [ %tmp10048, %bb86 ], [ %tmp100, %bb91 ]		; <i8*> [#uses=3]
	store i8 %tmp88, i8* %i.5.1
	%tmp109 = getelementptr i8* %tmp109.0, i32 -1		; <i8*> [#uses=3]
	%tmp112 = icmp ult i8* %tmp109, %tmp118		; <i1> [#uses=1]
	br i1 %tmp112, label %bb115, label %bb86

bb115:		; preds = %bb104, %cond_true82, %bb75, %bb46, %cond_true41, %bb37
	%tmp109.1 = phi i8* [ undef, %bb37 ], [ %tmp109.1, %bb75 ], [ %tmp10937, %cond_true82 ], [ %tmp109, %bb104 ], [ undef, %bb46 ], [ undef, %cond_true41 ]		; <i8*> [#uses=1]
	%i.5.3 = phi i8* [ undef, %bb37 ], [ %i.5.3, %bb75 ], [ %i.5.3, %cond_true82 ], [ %i.5.1, %bb104 ], [ undef, %bb46 ], [ undef, %cond_true41 ]		; <i8*> [#uses=3]
	%tmp100.3 = phi i8* [ undef, %bb37 ], [ %tmp100.3, %bb75 ], [ %tmp100.3, %cond_true82 ], [ %tmp100.1, %bb104 ], [ undef, %bb46 ], [ undef, %cond_true41 ]		; <i8*> [#uses=3]
	%min.1 = phi i8* [ %tmp118, %bb104 ], [ %tmp118, %bb75 ], [ %base, %bb37 ], [ %base, %bb46 ], [ %base, %cond_true41 ], [ %tmp118, %cond_true82 ]		; <i8*> [#uses=2]
	%j.5 = phi i8* [ %tmp100.1, %bb104 ], [ %j.5, %bb75 ], [ %tmp52, %bb46 ], [ %j.1.1, %bb37 ], [ %j.1.1, %cond_true41 ], [ %j.5, %cond_true82 ]		; <i8*> [#uses=2]
	%i.4 = phi i8* [ %i.5.1, %bb104 ], [ %i.4, %bb75 ], [ %tmp56, %bb46 ], [ undef, %bb37 ], [ %base, %cond_true41 ], [ %i.4, %cond_true82 ]		; <i8*> [#uses=2]
	%c.4 = phi i8 [ %tmp88, %bb104 ], [ %c.4, %bb75 ], [ %tmp.upgrd.3, %bb46 ], [ undef, %bb37 ], [ undef, %cond_true41 ], [ %c.4, %cond_true82 ]		; <i8> [#uses=2]
	%tmp116 = load i32* @qsz		; <i32> [#uses=2]
	%tmp118 = getelementptr i8* %min.1, i32 %tmp116		; <i8*> [#uses=9]
	%tmp122 = icmp ult i8* %tmp118, %tmp9		; <i1> [#uses=1]
	br i1 %tmp122, label %bb66, label %return

return:		; preds = %bb115, %entry
	ret i32 undef
}

declare i32 @qste(i8*, i8*)

declare i32 @comparee(%struct.edgeBox*, %struct.edgeBox*)
