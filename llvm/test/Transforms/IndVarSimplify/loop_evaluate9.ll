; RUN: opt < %s -indvars -S > %t
; RUN: grep "[%]tmp7 = icmp eq i8 -28, -28" %t
; RUN: grep "[%]tmp8 = icmp eq i8 63, 63" %t
; PR4477
; Indvars should compute the exit values in loop.
;
; XFAIL: *
; Indvars does not currently replace loop invariant values unless all
; loop exits have the same exit value. We could handle some cases,
; such as this, by making getSCEVAtScope() sensitive to a particular
; loop exit.  See PR11388.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
	%struct.cc70a02__complex_integers__complex_type = type { i8, i8 }
@.str = internal constant [13 x i8] c"fc70a00.adb\00\00", align 1		; <[13 x i8]*> [#uses=1]

define void @_ada_cc70a02() {
entry:
	br label %bb1.i

bb1.i:		; preds = %bb2.i, %entry
	%indvar.i = phi i32 [ 0, %entry ], [ %indvar.next.i, %bb2.i ]		; <i32> [#uses=2]
	%result.0.i = phi i16 [ 0, %entry ], [ %ins36.i, %bb2.i ]		; <i16> [#uses=2]
	%tmp38.i = trunc i16 %result.0.i to i8		; <i8> [#uses=2]
	%tmp = add i8 %tmp38.i, 96		; <i8> [#uses=1]
	%tmp1 = icmp ugt i8 %tmp, -56		; <i1> [#uses=1]
	br i1 %tmp1, label %bb.i.i, label %bb1.i.i

bb.i.i:		; preds = %bb1.i
	tail call void @__gnat_rcheck_12(i8* getelementptr ([13 x i8]* @.str, i32 0, i32 0), i32 24) noreturn
	unreachable

bb1.i.i:		; preds = %bb1.i
	%tmp41.i = lshr i16 %result.0.i, 8		; <i16> [#uses=1]
	%tmp42.i = trunc i16 %tmp41.i to i8		; <i8> [#uses=2]
	%tmp2 = add i8 %tmp42.i, 109		; <i8> [#uses=1]
	%tmp3 = icmp ugt i8 %tmp2, -56		; <i1> [#uses=1]
	br i1 %tmp3, label %bb2.i.i, label %cc70a02__complex_integers__Oadd.153.exit.i

bb2.i.i:		; preds = %bb1.i.i
	tail call void @__gnat_rcheck_12(i8* getelementptr ([13 x i8]* @.str, i32 0, i32 0), i32 24) noreturn
	unreachable

cc70a02__complex_integers__Oadd.153.exit.i:		; preds = %bb1.i.i
	%tmp4 = add i8 %tmp38.i, -4		; <i8> [#uses=2]
	%tmp5 = add i8 %tmp42.i, 9		; <i8> [#uses=2]
	%tmp25.i = zext i8 %tmp4 to i16		; <i16> [#uses=1]
	%tmp33.i = zext i8 %tmp5 to i16		; <i16> [#uses=1]
	%tmp34.i = shl i16 %tmp33.i, 8		; <i16> [#uses=1]
	%ins36.i = or i16 %tmp34.i, %tmp25.i		; <i16> [#uses=1]
	%tmp6 = icmp eq i32 %indvar.i, 6		; <i1> [#uses=1]
	br i1 %tmp6, label %cc70a02__complex_multiplication.170.exit, label %bb2.i

bb2.i:		; preds = %cc70a02__complex_integers__Oadd.153.exit.i
	%indvar.next.i = add i32 %indvar.i, 1		; <i32> [#uses=1]
	br label %bb1.i

cc70a02__complex_multiplication.170.exit:		; preds = %cc70a02__complex_integers__Oadd.153.exit.i
	%tmp7 = icmp eq i8 %tmp4, -28		; <i1> [#uses=1]
	%tmp8 = icmp eq i8 %tmp5, 63		; <i1> [#uses=1]
	%or.cond = and i1 %tmp8, %tmp7		; <i1> [#uses=1]
	br i1 %or.cond, label %return, label %bb1

bb1:		; preds = %cc70a02__complex_multiplication.170.exit
	tail call void @exit(i32 1)
	ret void

return:		; preds = %cc70a02__complex_multiplication.170.exit
	ret void
}

declare fastcc void @cc70a02__complex_integers__complex.164(%struct.cc70a02__complex_integers__complex_type* noalias nocapture sret, i8 signext, i8 signext) nounwind

declare fastcc void @cc70a02__complex_integers__Osubtract.149(%struct.cc70a02__complex_integers__complex_type* noalias sret, %struct.cc70a02__complex_integers__complex_type* byval align 4)

declare fastcc void @cc70a02__complex_integers__Oadd.153(%struct.cc70a02__complex_integers__complex_type* noalias sret, %struct.cc70a02__complex_integers__complex_type* byval align 4, %struct.cc70a02__complex_integers__complex_type* byval align 4)

declare fastcc void @cc70a02__complex_multiplication.170(%struct.cc70a02__complex_integers__complex_type* noalias sret, %struct.cc70a02__complex_integers__complex_type* byval align 4)

declare void @__gnat_rcheck_12(i8*, i32) noreturn

declare void @exit(i32)
