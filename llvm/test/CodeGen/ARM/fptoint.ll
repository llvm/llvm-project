; RUN: llvm-as < %s | llc -march=arm -mattr=+v6,+vfp2 | grep fmrs | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+v6,+vfp2 | not grep fmrrd

@i = weak global i32 0		; <i32*> [#uses=2]
@u = weak global i32 0		; <i32*> [#uses=2]

define i32 @foo1(float *%x) {
        %tmp1 = load float* %x
	%tmp2 = bitcast float %tmp1 to i32
	ret i32 %tmp2
}

define i64 @foo2(double *%x) {
        %tmp1 = load double* %x
	%tmp2 = bitcast double %tmp1 to i64
	ret i64 %tmp2
}

define void @foo5(float %x) {
	%tmp1 = fptosi float %x to i32
	store i32 %tmp1, i32* @i
	ret void
}

define void @foo6(float %x) {
	%tmp1 = fptoui float %x to i32
	store i32 %tmp1, i32* @u
	ret void
}

define void @foo7(double %x) {
	%tmp1 = fptosi double %x to i32
	store i32 %tmp1, i32* @i
	ret void
}

define void @foo8(double %x) {
	%tmp1 = fptoui double %x to i32
	store i32 %tmp1, i32* @u
	ret void
}

define void @foo9(double %x) {
	%tmp = fptoui double %x to i16
	store i16 %tmp, i16* null
	ret void
}
