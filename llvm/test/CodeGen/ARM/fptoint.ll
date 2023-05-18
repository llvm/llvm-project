; RUN: llc -mtriple=arm-eabi -arm-atomic-cfg-tidy=0 -mattr=+v6,+vfp2 %s -o - | FileCheck %s

@i = weak global i32 0		; <ptr> [#uses=2]
@u = weak global i32 0		; <ptr> [#uses=2]

define i32 @foo1(ptr %x) {
        %tmp1 = load float, ptr %x
	%tmp2 = bitcast float %tmp1 to i32
	ret i32 %tmp2
}

define i64 @foo2(ptr %x) {
        %tmp1 = load double, ptr %x
	%tmp2 = bitcast double %tmp1 to i64
	ret i64 %tmp2
}

define void @foo5(float %x) {
	%tmp1 = fptosi float %x to i32
	store i32 %tmp1, ptr @i
	ret void
}

define void @foo6(float %x) {
	%tmp1 = fptoui float %x to i32
	store i32 %tmp1, ptr @u
	ret void
}

define void @foo7(double %x) {
	%tmp1 = fptosi double %x to i32
	store i32 %tmp1, ptr @i
	ret void
}

define void @foo8(double %x) {
	%tmp1 = fptoui double %x to i32
	store i32 %tmp1, ptr @u
	ret void
}

define void @foo9(double %x) {
	%tmp = fptoui double %x to i16
	store i16 %tmp, ptr null
	ret void
}
; CHECK-LABEL: foo9:
; CHECK: 	vmov	r0, s0

