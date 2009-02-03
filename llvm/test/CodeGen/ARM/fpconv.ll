; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 > %t
; RUN: grep fcvtsd %t
; RUN: grep fcvtds %t
; RUN: grep ftosizs %t
; RUN: grep ftouizs %t
; RUN: grep ftosizd %t
; RUN: grep ftouizd %t
; RUN: grep fsitos %t
; RUN: grep fsitod %t
; RUN: grep fuitos %t
; RUN: grep fuitod %t
; RUN: llvm-as < %s | llc -march=arm > %t
; RUN: grep truncdfsf2 %t
; RUN: grep extendsfdf2 %t
; RUN: grep fixsfsi %t
; RUN: grep fixunssfsi %t
; RUN: grep fixdfsi %t
; RUN: grep fixunsdfsi %t
; RUN: grep floatsisf %t
; RUN: grep floatsidf %t
; RUN: grep floatunsisf %t
; RUN: grep floatunsidf %t
; RUN: llvm-as < %s | llc -march=thumb

define float @f1(double %x) {
entry:
	%tmp1 = fptrunc double %x to float		; <float> [#uses=1]
	ret float %tmp1
}

define double @f2(float %x) {
entry:
	%tmp1 = fpext float %x to double		; <double> [#uses=1]
	ret double %tmp1
}

define i32 @f3(float %x) {
entry:
	%tmp = fptosi float %x to i32		; <i32> [#uses=1]
	ret i32 %tmp
}

define i32 @f4(float %x) {
entry:
	%tmp = fptoui float %x to i32		; <i32> [#uses=1]
	ret i32 %tmp
}

define i32 @f5(double %x) {
entry:
	%tmp = fptosi double %x to i32		; <i32> [#uses=1]
	ret i32 %tmp
}

define i32 @f6(double %x) {
entry:
	%tmp = fptoui double %x to i32		; <i32> [#uses=1]
	ret i32 %tmp
}

define float @f7(i32 %a) {
entry:
	%tmp = sitofp i32 %a to float		; <float> [#uses=1]
	ret float %tmp
}

define double @f8(i32 %a) {
entry:
	%tmp = sitofp i32 %a to double		; <double> [#uses=1]
	ret double %tmp
}

define float @f9(i32 %a) {
entry:
	%tmp = uitofp i32 %a to float		; <float> [#uses=1]
	ret float %tmp
}

define double @f10(i32 %a) {
entry:
	%tmp = uitofp i32 %a to double		; <double> [#uses=1]
	ret double %tmp
}
