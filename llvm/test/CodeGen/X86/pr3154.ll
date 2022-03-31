; RUN: llc < %s -mtriple=i386-pc-linux-gnu -mattr=+sse2
; RUN: llc < %s -mtriple=i386-pc-linux-gnu -mattr=+sse2 -relocation-model=pic -frame-pointer=all
; PR3154

define void @ff_flac_compute_autocorr_sse2(i32* %data, i32 %len, i32 %lag, double* %autoc) nounwind {
entry:
	%c = alloca double, align 8		; <double*> [#uses=2]
	%0 = add i32 %len, 2		; <i32> [#uses=1]
	%1 = add i32 %0, %lag		; <i32> [#uses=1]
	%2 = alloca double, i32 %1		; <double*> [#uses=2]
	%3 = getelementptr double, double* %2, i32 %lag		; <double*> [#uses=2]
	%4 = ptrtoint double* %3 to i32		; <i32> [#uses=1]
	%5 = and i32 %4, 8		; <i32> [#uses=1]
	%6 = icmp eq i32 %5, 0		; <i1> [#uses=1]
	br i1 %6, label %bb19, label %bb

bb:		; preds = %entry
	%.sum = add i32 %lag, 1		; <i32> [#uses=1]
	%7 = getelementptr double, double* %2, i32 %.sum		; <double*> [#uses=1]
	br label %bb19

bb19:		; preds = %bb, %entry
	%data15.0 = phi double* [ %7, %bb ], [ %3, %entry ]		; <double*> [#uses=5]
	%8 = sitofp i32 %len to double		; <double> [#uses=1]
	%9 = fsub double %8, 1.000000e+00		; <double> [#uses=1]
	%10 = fdiv double 2.000000e+00, %9		; <double> [#uses=1]
	store double %10, double* %c, align 8
	%11 = ashr i32 %len, 1		; <i32> [#uses=3]
	%12 = mul i32 %11, -4		; <i32> [#uses=2]
	%13 = shl i32 %len, 1		; <i32> [#uses=1]
	%14 = and i32 %13, -4		; <i32> [#uses=2]
	call void asm sideeffect "movsd   $0,     %xmm7                \0A\09movapd  ff_pd_1, %xmm6     \0A\09movapd  ff_pd_2, %xmm5     \0A\09movlhps %xmm7, %xmm7                \0A\09subpd   %xmm5, %xmm7                \0A\09addsd   %xmm6, %xmm7                \0A\09", "*m,~{dirflag},~{fpsr},~{flags}"(double* elementtype(double) %c) nounwind
	%15 = and i32 %len, 1		; <i32> [#uses=1]
	%toBool = icmp eq i32 %15, 0		; <i1> [#uses=1]
	%16 = getelementptr double, double* %data15.0, i32 %11		; <double*> [#uses=2]
	%17 = getelementptr i32, i32* %data, i32 %11		; <i32*> [#uses=2]
	br i1 %toBool, label %bb22, label %bb20

bb20:		; preds = %bb19
	%asmtmp = call { i32, i32 } asm sideeffect "1:                                    \0A\09movapd   %xmm7,  %xmm1              \0A\09mulpd    %xmm1,  %xmm1              \0A\09movapd   %xmm6,  %xmm0              \0A\09subpd    %xmm1,  %xmm0              \0A\09pshufd   $$0x4e,   %xmm0, %xmm1      \0A\09cvtpi2pd ($3,$0), %xmm2              \0A\09cvtpi2pd -1*4($3,$1), %xmm3   \0A\09mulpd    %xmm0,  %xmm2              \0A\09mulpd    %xmm1,  %xmm3              \0A\09movapd   %xmm2, ($2,$0,2)            \0A\09movupd    %xmm3, -1*8($2,$1,2) \0A\09subpd    %xmm5,  %xmm7              \0A\09sub      $$8,      $1                  \0A\09add      $$8,      $0                  \0A\09jl 1b                                 \0A\09", "=&r,=&r,r,r,0,1,~{dirflag},~{fpsr},~{flags}"(double* %16, i32* %17, i32 %12, i32 %14) nounwind		; <{ i32, i32 }> [#uses=0]
	br label %bb28.preheader

bb22:		; preds = %bb19
	%asmtmp23 = call { i32, i32 } asm sideeffect "1:                                    \0A\09movapd   %xmm7,  %xmm1              \0A\09mulpd    %xmm1,  %xmm1              \0A\09movapd   %xmm6,  %xmm0              \0A\09subpd    %xmm1,  %xmm0              \0A\09pshufd   $$0x4e,   %xmm0, %xmm1      \0A\09cvtpi2pd ($3,$0), %xmm2              \0A\09cvtpi2pd -2*4($3,$1), %xmm3   \0A\09mulpd    %xmm0,  %xmm2              \0A\09mulpd    %xmm1,  %xmm3              \0A\09movapd   %xmm2, ($2,$0,2)            \0A\09movapd    %xmm3, -2*8($2,$1,2) \0A\09subpd    %xmm5,  %xmm7              \0A\09sub      $$8,      $1                  \0A\09add      $$8,      $0                  \0A\09jl 1b                                 \0A\09", "=&r,=&r,r,r,0,1,~{dirflag},~{fpsr},~{flags}"(double* %16, i32* %17, i32 %12, i32 %14) nounwind		; <{ i32, i32 }> [#uses=0]
	br label %bb28.preheader

bb28.preheader:		; preds = %bb22, %bb20
	%18 = icmp sgt i32 %lag, 0		; <i1> [#uses=2]
	br i1 %18, label %bb27, label %bb29

bb27:		; preds = %bb27, %bb28.preheader
	%j4.042 = phi i32 [ 0, %bb28.preheader ], [ %indvar.next45, %bb27 ]		; <i32> [#uses=2]
	%19 = sub i32 %j4.042, %lag		; <i32> [#uses=1]
	%20 = getelementptr double, double* %data15.0, i32 %19		; <double*> [#uses=1]
	store double 0.000000e+00, double* %20, align 8
	%indvar.next45 = add i32 %j4.042, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next45, %lag		; <i1> [#uses=1]
	br i1 %exitcond, label %bb29, label %bb27

bb29:		; preds = %bb27, %bb28.preheader
	%21 = getelementptr double, double* %data15.0, i32 %len		; <double*> [#uses=3]
	store double 0.000000e+00, double* %21, align 8
	br i1 %18, label %bb.nph, label %bb37

bb.nph:		; preds = %bb29
	%22 = mul i32 %len, -8		; <i32> [#uses=2]
	%23 = add i32 %lag, -2		; <i32> [#uses=1]
	br label %bb30

bb30:		; preds = %bb35, %bb.nph
	%indvar = phi i32 [ 0, %bb.nph ], [ %indvar.next, %bb35 ]		; <i32> [#uses=2]
	%j4.141 = shl i32 %indvar, 1		; <i32> [#uses=8]
	%24 = icmp eq i32 %23, %j4.141		; <i1> [#uses=1]
	%25 = or i32 %j4.141, 1		; <i32> [#uses=2]
	br i1 %24, label %bb31, label %bb33

bb31:		; preds = %bb30
	%26 = add i32 %j4.141, 2		; <i32> [#uses=2]
	%.sum38 = sub i32 %len, %j4.141		; <i32> [#uses=1]
	%27 = getelementptr double, double* %data15.0, i32 %.sum38		; <double*> [#uses=1]
	%28 = getelementptr double, double* %autoc, i32 %j4.141		; <double*> [#uses=1]
	%29 = getelementptr double, double* %autoc, i32 %25		; <double*> [#uses=1]
	%30 = getelementptr double, double* %autoc, i32 %26		; <double*> [#uses=1]
	%asmtmp32 = call i32 asm sideeffect "movsd    ff_pd_1, %xmm0 \0A\09movsd    ff_pd_1, %xmm1 \0A\09movsd    ff_pd_1, %xmm2 \0A\091:                                 \0A\09movapd   ($4,$0), %xmm3           \0A\09movupd -8($5,$0), %xmm4           \0A\09movapd   ($5,$0), %xmm5           \0A\09mulpd     %xmm3, %xmm4           \0A\09mulpd     %xmm3, %xmm5           \0A\09mulpd -16($5,$0), %xmm3           \0A\09addpd     %xmm4, %xmm1           \0A\09addpd     %xmm5, %xmm0           \0A\09addpd     %xmm3, %xmm2           \0A\09add       $$16,    $0               \0A\09jl 1b                              \0A\09movhlps   %xmm0, %xmm3           \0A\09movhlps   %xmm1, %xmm4           \0A\09movhlps   %xmm2, %xmm5           \0A\09addsd     %xmm3, %xmm0           \0A\09addsd     %xmm4, %xmm1           \0A\09addsd     %xmm5, %xmm2           \0A\09movsd     %xmm0, $1               \0A\09movsd     %xmm1, $2               \0A\09movsd     %xmm2, $3               \0A\09", "=&r,=*m,=*m,=*m,r,r,0,~{dirflag},~{fpsr},~{flags}"(double* elementtype(double) %28, double* elementtype(double) %29, double* elementtype(double) %30, double* %21, double* %27, i32 %22) nounwind		; <i32> [#uses=0]
	br label %bb35

bb33:		; preds = %bb30
	%.sum39 = sub i32 %len, %j4.141		; <i32> [#uses=1]
	%31 = getelementptr double, double* %data15.0, i32 %.sum39		; <double*> [#uses=1]
	%32 = getelementptr double, double* %autoc, i32 %j4.141		; <double*> [#uses=1]
	%33 = getelementptr double, double* %autoc, i32 %25		; <double*> [#uses=1]
	%asmtmp34 = call i32 asm sideeffect "movsd    ff_pd_1, %xmm0 \0A\09movsd    ff_pd_1, %xmm1 \0A\091:                                 \0A\09movapd   ($3,$0), %xmm3           \0A\09movupd -8($4,$0), %xmm4           \0A\09mulpd     %xmm3, %xmm4           \0A\09mulpd    ($4,$0), %xmm3           \0A\09addpd     %xmm4, %xmm1           \0A\09addpd     %xmm3, %xmm0           \0A\09add       $$16,    $0               \0A\09jl 1b                              \0A\09movhlps   %xmm0, %xmm3           \0A\09movhlps   %xmm1, %xmm4           \0A\09addsd     %xmm3, %xmm0           \0A\09addsd     %xmm4, %xmm1           \0A\09movsd     %xmm0, $1               \0A\09movsd     %xmm1, $2               \0A\09", "=&r,=*m,=*m,r,r,0,~{dirflag},~{fpsr},~{flags}"(double* elementtype(double) %32, double* elementtype(double) %33, double* %21, double* %31, i32 %22) nounwind		; <i32> [#uses=0]
	%.pre = add i32 %j4.141, 2		; <i32> [#uses=1]
	br label %bb35

bb35:		; preds = %bb33, %bb31
	%.pre-phi = phi i32 [ %.pre, %bb33 ], [ %26, %bb31 ]		; <i32> [#uses=1]
	%34 = icmp slt i32 %.pre-phi, %lag		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %34, label %bb30, label %bb37

bb37:		; preds = %bb35, %bb29
	ret void
}
