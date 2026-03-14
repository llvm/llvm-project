; RUN: llc < %s -mtriple=arm-apple-darwin

	%struct.Decoders = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr }
@decoders = external global %struct.Decoders		; <ptr> [#uses=1]

declare ptr @calloc(i32, i32)

declare fastcc i32 @get_mem2Dint(ptr, i32, i32)

define fastcc void @init_global_buffers() nounwind {
entry:
	%tmp151 = tail call fastcc i32 @get_mem2Dint( ptr @decoders, i32 16, i32 16 )		; <i32> [#uses=1]
	%tmp158 = tail call ptr @calloc( i32 0, i32 4 )		; <ptr> [#uses=0]
	br i1 false, label %cond_true166, label %bb190.preheader

bb190.preheader:		; preds = %entry
	%memory_size.3555 = add i32 0, %tmp151		; <i32> [#uses=0]
	unreachable

cond_true166:		; preds = %entry
	unreachable
}
