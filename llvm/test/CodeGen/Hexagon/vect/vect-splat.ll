; Extracted from test/CodeGen/Generic/vector.ll: used to loop indefinitely.
; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
; CHECK: splat_i4

%i4 = type <4 x i32>

define void @splat_i4(ptr %P, ptr %Q, i32 %X) {
	%tmp = insertelement %i4 undef, i32 %X, i32 0		; <%i4> [#uses=1]
	%tmp2 = insertelement %i4 %tmp, i32 %X, i32 1		; <%i4> [#uses=1]
	%tmp4 = insertelement %i4 %tmp2, i32 %X, i32 2		; <%i4> [#uses=1]
	%tmp6 = insertelement %i4 %tmp4, i32 %X, i32 3		; <%i4> [#uses=1]
	%q = load %i4, ptr %Q		; <%i4> [#uses=1]
	%R = add %i4 %q, %tmp6		; <%i4> [#uses=1]
	store %i4 %R, ptr %P
	ret void
}
