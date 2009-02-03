; RUN: llvm-as < %s | opt -scalarrepl -disable-output

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64"
target triple = "arm-apple-darwin8"
	%struct.CGPoint = type { float, float }
	%struct.aal_big_range_t = type { i32, i32 }
	%struct.aal_callback_t = type { i8* (i8*, i32)*, void (i8*, i8*)* }
	%struct.aal_edge_pool_t = type { %struct.aal_edge_pool_t*, i32, i32, [0 x %struct.aal_edge_t] }
	%struct.aal_edge_t = type { %struct.CGPoint, %struct.CGPoint, i32 }
	%struct.aal_range_t = type { i16, i16 }
	%struct.aal_span_pool_t = type { %struct.aal_span_pool_t*, [341 x %struct.aal_span_t] }
	%struct.aal_span_t = type { %struct.aal_span_t*, %struct.aal_big_range_t }
	%struct.aal_spanarray_t = type { [2 x %struct.aal_range_t] }
	%struct.aal_spanbucket_t = type { i16, [2 x i8], %struct.anon }
	%struct.aal_state_t = type { %struct.CGPoint, %struct.CGPoint, %struct.CGPoint, i32, float, float, float, float, %struct.CGPoint, %struct.CGPoint, float, float, float, float, i32, i32, i32, i32, float, float, i8*, i32, i32, %struct.aal_edge_pool_t*, %struct.aal_edge_pool_t*, i8*, %struct.aal_callback_t*, i32, %struct.aal_span_t*, %struct.aal_span_t*, %struct.aal_span_t*, %struct.aal_span_pool_t*, i8, float, i8, i32 }
	%struct.anon = type { %struct.aal_spanarray_t }


declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

define fastcc void @aal_insert_span() {
entry:
	%SB = alloca %struct.aal_spanbucket_t, align 4		; <%struct.aal_spanbucket_t*> [#uses=2]
	br i1 false, label %cond_true, label %cond_next79

cond_true:		; preds = %entry
	br i1 false, label %cond_next, label %cond_next114.i

cond_next114.i:		; preds = %cond_true
	ret void

cond_next:		; preds = %cond_true
	%SB19 = bitcast %struct.aal_spanbucket_t* %SB to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %SB19, i8* null, i32 12, i32 0 )
	br i1 false, label %cond_next34, label %cond_next79

cond_next34:		; preds = %cond_next
	%i.2.reload22 = load i32* null		; <i32> [#uses=1]
	%tmp51 = getelementptr %struct.aal_spanbucket_t* %SB, i32 0, i32 2, i32 0, i32 0, i32 %i.2.reload22, i32 1		; <i16*> [#uses=0]
	ret void

cond_next79:		; preds = %cond_next, %entry
	ret void
}
