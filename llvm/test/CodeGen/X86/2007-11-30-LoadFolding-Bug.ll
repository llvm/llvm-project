; RUN: llc < %s -mtriple=i686-- -mattr=+sse2 | FileCheck %s
; Increment in loop bb.i28.i adjusted to 2, to prevent loop reversal from
; kicking in.

declare fastcc void @rdft(i32, i32, double*, i32*, double*)

define fastcc void @mp_sqrt(i32 %n, i32 %radix, i32* %in, i32* %out, i32* %tmp1, i32* %tmp2, i32 %nfft, double* %tmp1fft, double* %tmp2fft, i32* %ip, double* %w) nounwind {
entry:
	br label %bb.i5

bb.i5:		; preds = %bb.i5, %entry
	%nfft_init.0.i = phi i32 [ 1, %entry ], [ %tmp7.i3, %bb.i5 ]		; <i32> [#uses=1]
	%foo = phi i1 [1, %entry], [0, %bb.i5]
	%tmp7.i3 = shl i32 %nfft_init.0.i, 1		; <i32> [#uses=2]
	br i1 %foo, label %bb.i5, label %mp_unexp_mp2d.exit.i

mp_unexp_mp2d.exit.i:		; preds = %bb.i5
	br i1 %foo, label %cond_next.i, label %cond_true.i

cond_true.i:		; preds = %mp_unexp_mp2d.exit.i
	ret void

cond_next.i:		; preds = %mp_unexp_mp2d.exit.i
	%tmp22.i = sdiv i32 0, 2		; <i32> [#uses=2]
	br i1 %foo, label %cond_true29.i, label %cond_next36.i

cond_true29.i:		; preds = %cond_next.i
	ret void

cond_next36.i:		; preds = %cond_next.i
	store i32 %tmp22.i, i32* null, align 4
	%tmp8.i14.i = select i1 %foo, i32 1, i32 0		; <i32> [#uses=1]
	br label %bb.i28.i

bb.i28.i:		; preds = %bb.i28.i, %cond_next36.i
; CHECK: %bb.i28.i
; CHECK: addl $2
; CHECK: addl $-2
	%j.0.reg2mem.0.i16.i = phi i32 [ 0, %cond_next36.i ], [ %indvar.next39.i, %bb.i28.i ]		; <i32> [#uses=2]
	%din_addr.1.reg2mem.0.i17.i = phi double [ 0.000000e+00, %cond_next36.i ], [ %tmp16.i25.i, %bb.i28.i ]		; <double> [#uses=1]
	%tmp1.i18.i = fptosi double %din_addr.1.reg2mem.0.i17.i to i32		; <i32> [#uses=2]
	%tmp4.i19.i = icmp slt i32 %tmp1.i18.i, %radix		; <i1> [#uses=1]
	%x.0.i21.i = select i1 %tmp4.i19.i, i32 %tmp1.i18.i, i32 0		; <i32> [#uses=1]
	%tmp41.sum.i = add i32 %j.0.reg2mem.0.i16.i, 2		; <i32> [#uses=0]
	%tmp1213.i23.i = sitofp i32 %x.0.i21.i to double		; <double> [#uses=1]
	%tmp15.i24.i = fsub double 0.000000e+00, %tmp1213.i23.i		; <double> [#uses=1]
	%tmp16.i25.i = fmul double 0.000000e+00, %tmp15.i24.i		; <double> [#uses=1]
	%indvar.next39.i = add i32 %j.0.reg2mem.0.i16.i, 2		; <i32> [#uses=2]
	%exitcond40.i = icmp eq i32 %indvar.next39.i, %tmp8.i14.i		; <i1> [#uses=1]
	br i1 %exitcond40.i, label %mp_unexp_d2mp.exit29.i, label %bb.i28.i

mp_unexp_d2mp.exit29.i:		; preds = %bb.i28.i
	%tmp46.i = sub i32 0, %tmp22.i		; <i32> [#uses=1]
	store i32 %tmp46.i, i32* null, align 4
	br i1 %exitcond40.i, label %bb.i.i, label %mp_sqrt_init.exit

bb.i.i:		; preds = %bb.i.i, %mp_unexp_d2mp.exit29.i
	br label %bb.i.i

mp_sqrt_init.exit:		; preds = %mp_unexp_d2mp.exit29.i
	tail call fastcc void @mp_mul_csqu( i32 0, double* %tmp1fft )
	tail call fastcc void @rdft( i32 0, i32 -1, double* null, i32* %ip, double* %w )
	tail call fastcc void @mp_mul_d2i( i32 0, i32 %radix, i32 0, double* %tmp1fft, i32* %tmp2 )
	br i1 %exitcond40.i, label %cond_false.i, label %cond_true36.i

cond_true36.i:		; preds = %mp_sqrt_init.exit
	ret void

cond_false.i:		; preds = %mp_sqrt_init.exit
	tail call fastcc void @mp_round( i32 0, i32 %radix, i32 0, i32* %out )
	tail call fastcc void @mp_add( i32 0, i32 %radix, i32* %tmp1, i32* %tmp2, i32* %tmp1 )
	tail call fastcc void @mp_sub( i32 0, i32 %radix, i32* %in, i32* %tmp2, i32* %tmp2 )
	tail call fastcc void @mp_round( i32 0, i32 %radix, i32 0, i32* %tmp1 )
	tail call fastcc void @mp_mul_d2i( i32 0, i32 %radix, i32 %tmp7.i3, double* %tmp2fft, i32* %tmp2 )
	ret void
}

declare fastcc void @mp_add(i32, i32, i32*, i32*, i32*)

declare fastcc void @mp_sub(i32, i32, i32*, i32*, i32*)

declare fastcc void @mp_round(i32, i32, i32, i32*)

declare fastcc void @mp_mul_csqu(i32, double*)

declare fastcc void @mp_mul_d2i(i32, i32, i32, double*, i32*)
