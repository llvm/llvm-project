; RUN: llvm-as < %s | opt -memcpyopt | llvm-dis | not grep store
; RUN: llvm-as < %s | opt -memcpyopt | llvm-dis | grep {call.*llvm.memset} | count 3

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
	%struct.MV = type { i16, i16 }

define i32 @t() nounwind  {
entry:
	%ref_idx = alloca [8 x i8]		; <[8 x i8]*> [#uses=8]
	%left_mvd = alloca [8 x %struct.MV]		; <[8 x %struct.MV]*> [#uses=17]
	%up_mvd = alloca [8 x %struct.MV]		; <[8 x %struct.MV]*> [#uses=17]
	%tmp20 = getelementptr [8 x i8]* %ref_idx, i32 0, i32 7		; <i8*> [#uses=1]
	store i8 -1, i8* %tmp20, align 1
	%tmp23 = getelementptr [8 x i8]* %ref_idx, i32 0, i32 6		; <i8*> [#uses=1]
	store i8 -1, i8* %tmp23, align 1
	%tmp26 = getelementptr [8 x i8]* %ref_idx, i32 0, i32 5		; <i8*> [#uses=1]
	store i8 -1, i8* %tmp26, align 1
	%tmp29 = getelementptr [8 x i8]* %ref_idx, i32 0, i32 4		; <i8*> [#uses=1]
	store i8 -1, i8* %tmp29, align 1
	%tmp32 = getelementptr [8 x i8]* %ref_idx, i32 0, i32 3		; <i8*> [#uses=1]
	store i8 -1, i8* %tmp32, align 1
	%tmp35 = getelementptr [8 x i8]* %ref_idx, i32 0, i32 2		; <i8*> [#uses=1]
	store i8 -1, i8* %tmp35, align 1
	%tmp38 = getelementptr [8 x i8]* %ref_idx, i32 0, i32 1		; <i8*> [#uses=1]
	store i8 -1, i8* %tmp38, align 1
	%tmp41 = getelementptr [8 x i8]* %ref_idx, i32 0, i32 0		; <i8*> [#uses=2]
	store i8 -1, i8* %tmp41, align 1
	%tmp43 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 7, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp43, align 2
	%tmp46 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 7, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp46, align 2
	%tmp57 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 6, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp57, align 2
	%tmp60 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 6, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp60, align 2
	%tmp71 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 5, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp71, align 2
	%tmp74 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 5, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp74, align 2
	%tmp85 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 4, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp85, align 2
	%tmp88 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 4, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp88, align 2
	%tmp99 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 3, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp99, align 2
	%tmp102 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 3, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp102, align 2
	%tmp113 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 2, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp113, align 2
	%tmp116 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 2, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp116, align 2
	%tmp127 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 1, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp127, align 2
	%tmp130 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 1, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp130, align 2
	%tmp141 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 0, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp141, align 8
	%tmp144 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 0, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp144, align 2
	%tmp148 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 7, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp148, align 2
	%tmp151 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 7, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp151, align 2
	%tmp162 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 6, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp162, align 2
	%tmp165 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 6, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp165, align 2
	%tmp176 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 5, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp176, align 2
	%tmp179 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 5, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp179, align 2
	%tmp190 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 4, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp190, align 2
	%tmp193 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 4, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp193, align 2
	%tmp204 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 3, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp204, align 2
	%tmp207 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 3, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp207, align 2
	%tmp218 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 2, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp218, align 2
	%tmp221 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 2, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp221, align 2
	%tmp232 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 1, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp232, align 2
	%tmp235 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 1, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp235, align 2
	%tmp246 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 0, i32 0		; <i16*> [#uses=1]
	store i16 0, i16* %tmp246, align 8
	%tmp249 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 0, i32 1		; <i16*> [#uses=1]
	store i16 0, i16* %tmp249, align 2
	%up_mvd252 = getelementptr [8 x %struct.MV]* %up_mvd, i32 0, i32 0		; <%struct.MV*> [#uses=1]
	%left_mvd253 = getelementptr [8 x %struct.MV]* %left_mvd, i32 0, i32 0		; <%struct.MV*> [#uses=1]
	call void @foo( %struct.MV* %up_mvd252, %struct.MV* %left_mvd253, i8* %tmp41 ) nounwind 
	ret i32 undef
}

declare void @foo(%struct.MV*, %struct.MV*, i8*)
