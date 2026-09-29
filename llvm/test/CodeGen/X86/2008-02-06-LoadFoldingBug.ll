; RUN: llc < %s -mtriple=i686-- -mattr=+sse2 | FileCheck %s

; CHECK: xorps {{.*}}{{LCPI0_0|__xmm@}}
define void @casin(ptr sret({ double, double })  %agg.result, double %z.0, double %z.1) nounwind  {
entry:
	%memtmp = alloca { double, double }, align 8		; <ptr> [#uses=3]
	%tmp4 = fsub double -0.000000e+00, %z.1		; <double> [#uses=1]
	call void @casinh( ptr sret({ double, double })  %memtmp, double %tmp4, double %z.0 ) nounwind
	%tmp19 = getelementptr { double, double }, ptr %memtmp, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp20 = load double, ptr %tmp19, align 8		; <double> [#uses=1]
	%tmp22 = getelementptr { double, double }, ptr %memtmp, i32 0, i32 1		; <ptr> [#uses=1]
	%tmp23 = load double, ptr %tmp22, align 8		; <double> [#uses=1]
	%tmp32 = fsub double -0.000000e+00, %tmp20		; <double> [#uses=1]
	%tmp37 = getelementptr { double, double }, ptr %agg.result, i32 0, i32 0		; <ptr> [#uses=1]
	store double %tmp23, ptr %tmp37, align 8
	%tmp40 = getelementptr { double, double }, ptr %agg.result, i32 0, i32 1		; <ptr> [#uses=1]
	store double %tmp32, ptr %tmp40, align 8
	ret void
}

declare void @casinh(ptr sret({ double, double }) , double, double) nounwind
