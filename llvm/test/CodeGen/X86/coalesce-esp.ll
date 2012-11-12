; RUN: llc < %s | grep "movl	%esp, %ebp"
; PR4572

; Don't coalesce with %esp if it would end up putting %esp in
; the index position of an address, because that can't be
; encoded on x86. It would actually be slightly better to
; swap the address operands though, since there's no scale.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-mingw32"
	%"struct.std::valarray<unsigned int>" = type { i32, i32* }

define void @_ZSt17__gslice_to_indexjRKSt8valarrayIjES2_RS0_(i32 %__o, %"struct.std::valarray<unsigned int>"* nocapture %__l, %"struct.std::valarray<unsigned int>"* nocapture %__s, %"struct.std::valarray<unsigned int>"* nocapture %__i) nounwind {
entry:
	%0 = alloca i32, i32 undef, align 4		; <i32*> [#uses=1]
	br i1 undef, label %return, label %bb4

bb4:		; preds = %bb7.backedge, %entry
	%indvar = phi i32 [ %indvar.next, %bb7.backedge ], [ 0, %entry ]		; <i32> [#uses=2]
	%scevgep24.sum = sub i32 undef, %indvar		; <i32> [#uses=2]
	%scevgep25 = getelementptr i32* %0, i32 %scevgep24.sum		; <i32*> [#uses=1]
	%scevgep27 = getelementptr i32* undef, i32 %scevgep24.sum		; <i32*> [#uses=1]
	%1 = load i32* %scevgep27, align 4		; <i32> [#uses=0]
	br i1 undef, label %bb7.backedge, label %bb5

bb5:		; preds = %bb4
	store i32 0, i32* %scevgep25, align 4
	br label %bb7.backedge

bb7.backedge:		; preds = %bb5, %bb4
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br label %bb4

return:		; preds = %entry
	ret void
}
