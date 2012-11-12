; RUN: llc < %s -mtriple=arm-apple-darwin

	%struct.Decoders = type { i32**, i16***, i16****, i16***, i16**, i8**, i8** }
@decoders = external global %struct.Decoders		; <%struct.Decoders*> [#uses=1]

declare i8* @calloc(i32, i32)

declare fastcc i32 @get_mem2Dint(i32***, i32, i32)

define fastcc void @init_global_buffers() nounwind {
entry:
	%tmp151 = tail call fastcc i32 @get_mem2Dint( i32*** getelementptr (%struct.Decoders* @decoders, i32 0, i32 0), i32 16, i32 16 )		; <i32> [#uses=1]
	%tmp158 = tail call i8* @calloc( i32 0, i32 4 )		; <i8*> [#uses=0]
	br i1 false, label %cond_true166, label %bb190.preheader

bb190.preheader:		; preds = %entry
	%memory_size.3555 = add i32 0, %tmp151		; <i32> [#uses=0]
	unreachable

cond_true166:		; preds = %entry
	unreachable
}
