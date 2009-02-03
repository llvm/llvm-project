; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2
	%struct.foo = type opaque

define fastcc i32 @test(%struct.foo* %v, %struct.foo* %vi) {
	br i1 false, label %ilog2.exit, label %cond_true.i

cond_true.i:		; preds = %0
	ret i32 0

ilog2.exit:		; preds = %0
	%tmp24.i = load i32* null		; <i32> [#uses=1]
	%tmp13.i12.i = tail call double @ldexp( double 0.000000e+00, i32 0 )		; <double> [#uses=1]
	%tmp13.i13.i = fptrunc double %tmp13.i12.i to float		; <float> [#uses=1]
	%tmp11.s = load i32* null		; <i32> [#uses=1]
	%tmp11.i = bitcast i32 %tmp11.s to i32		; <i32> [#uses=1]
	%n.i = bitcast i32 %tmp24.i to i32		; <i32> [#uses=1]
	%tmp13.i7 = mul i32 %tmp11.i, %n.i		; <i32> [#uses=1]
	%tmp.i8 = tail call i8* @calloc( i32 %tmp13.i7, i32 4 )		; <i8*> [#uses=0]
	br i1 false, label %bb224.preheader.i, label %bb.i

bb.i:		; preds = %ilog2.exit
	ret i32 0

bb224.preheader.i:		; preds = %ilog2.exit
	%tmp165.i = fpext float %tmp13.i13.i to double		; <double> [#uses=0]
	ret i32 0
}

declare i8* @calloc(i32, i32)

declare double @ldexp(double, i32)
