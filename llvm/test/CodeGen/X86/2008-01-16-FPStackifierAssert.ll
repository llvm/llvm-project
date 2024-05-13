; RUN: llc < %s -mtriple=i686-- -mattr=+sse2 -regalloc=fast -optimize-regalloc=0

define void @SolveCubic(double %a, double %b, double %c, double %d, ptr %solutions, ptr %x) {
entry:
	%tmp71 = load x86_fp80, ptr null, align 16		; <x86_fp80> [#uses=1]
	%tmp72 = fdiv x86_fp80 %tmp71, 0xKC000C000000000000000		; <x86_fp80> [#uses=1]
	%tmp73 = fadd x86_fp80 0xK00000000000000000000, %tmp72		; <x86_fp80> [#uses=1]
	%tmp7374 = fptrunc x86_fp80 %tmp73 to double		; <double> [#uses=1]
	store double %tmp7374, ptr null, align 8
	%tmp81 = load double, ptr null, align 8		; <double> [#uses=1]
	%tmp82 = fadd double %tmp81, 0x401921FB54442D18		; <double> [#uses=1]
	%tmp83 = fdiv double %tmp82, 3.000000e+00		; <double> [#uses=1]
	%tmp84 = call double @cos( double %tmp83 )		; <double> [#uses=1]
	%tmp85 = fmul double 0.000000e+00, %tmp84		; <double> [#uses=1]
	%tmp8586 = fpext double %tmp85 to x86_fp80		; <x86_fp80> [#uses=1]
	%tmp87 = load x86_fp80, ptr null, align 16		; <x86_fp80> [#uses=1]
	%tmp88 = fdiv x86_fp80 %tmp87, 0xKC000C000000000000000		; <x86_fp80> [#uses=1]
	%tmp89 = fadd x86_fp80 %tmp8586, %tmp88		; <x86_fp80> [#uses=1]
	%tmp8990 = fptrunc x86_fp80 %tmp89 to double		; <double> [#uses=1]
	store double %tmp8990, ptr null, align 8
	%tmp97 = load double, ptr null, align 8		; <double> [#uses=1]
	%tmp98 = fadd double %tmp97, 0x402921FB54442D18		; <double> [#uses=1]
	%tmp99 = fdiv double %tmp98, 3.000000e+00		; <double> [#uses=1]
	%tmp100 = call double @cos( double %tmp99 )		; <double> [#uses=1]
	%tmp101 = fmul double 0.000000e+00, %tmp100		; <double> [#uses=1]
	%tmp101102 = fpext double %tmp101 to x86_fp80		; <x86_fp80> [#uses=1]
	%tmp103 = load x86_fp80, ptr null, align 16		; <x86_fp80> [#uses=1]
	%tmp104 = fdiv x86_fp80 %tmp103, 0xKC000C000000000000000		; <x86_fp80> [#uses=1]
	%tmp105 = fadd x86_fp80 %tmp101102, %tmp104		; <x86_fp80> [#uses=1]
	%tmp105106 = fptrunc x86_fp80 %tmp105 to double		; <double> [#uses=1]
	store double %tmp105106, ptr null, align 8
	ret void
}

declare double @cos(double)
