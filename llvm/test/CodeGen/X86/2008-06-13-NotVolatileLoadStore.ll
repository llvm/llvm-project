; RUN: llc < %s -mtriple=i686-- -fixup-byte-word-insts=0 | FileCheck %s -check-prefix=CHECK -check-prefix=BWOFF
; RUN: llc < %s -mtriple=i686-- -fixup-byte-word-insts=1 | FileCheck %s -check-prefix=CHECK -check-prefix=BWON
; These transforms are turned off for load volatiles and stores.
; Check that they weren't turned off for all loads and stores!
; CHECK-LABEL: f:
; CHECK-NOT: movsd
; BWOFF: movw
; BWON:  movzwl
; CHECK: addw

@atomic = global double 0.000000e+00		; <ptr> [#uses=1]
@atomic2 = global double 0.000000e+00		; <ptr> [#uses=1]
@ioport = global i32 0		; <ptr> [#uses=1]
@ioport2 = global i32 0		; <ptr> [#uses=1]

define i16 @f(i64 %x) {
	%b = bitcast i64 %x to double		; <double> [#uses=1]
	store double %b, ptr @atomic
	store double 0.000000e+00, ptr @atomic2
	%l = load i32, ptr @ioport		; <i32> [#uses=1]
	%t = trunc i32 %l to i16		; <i16> [#uses=1]
	%l2 = load i32, ptr @ioport2		; <i32> [#uses=1]
	%tmp = lshr i32 %l2, 16		; <i32> [#uses=1]
	%t2 = trunc i32 %tmp to i16		; <i16> [#uses=1]
	%f = add i16 %t, %t2		; <i16> [#uses=1]
	ret i16 %f
}
