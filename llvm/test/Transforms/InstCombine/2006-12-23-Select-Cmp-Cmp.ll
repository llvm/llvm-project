; For PR1065. This causes an assertion in instcombine if a select with two cmp
; operands is encountered.
; RUN: opt < %s -passes=instcombine -disable-output
; END.

target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.internal_state = type { i32 }
	%struct.mng_data = type { i32, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i32, i32, i32, i8, i32, i32, i32, i32, i16, i16, i16, i8, i8, double, double, double, i8, i8, i8, i8, i32, i32, i32, i32, i32, i8, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i8, i32, i32, ptr, ptr, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i8, i8, i8, i8, i8, i32, i8, i8, i8, i32, ptr, i32, ptr, i32, i8, i8, i8, i32, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, i8, i8, i32, i32, ptr, i8, i8, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8, i32, i32, ptr, i32, i32, i32, i8, i8, i32, i32, i32, i32, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, i8, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, [256 x i8], double, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i16, i8, i8, i8, i8, i8, i32, i32, i8, i32, i32, i32, i32, i16, i16, i16, i8, i16, i8, i32, i32, i32, i32, i8, i32, i32, i8, i32, i32, i32, i32, i8, i32, i32, i8, i32, i32, i32, i32, i32, i8, i32, i8, i16, i16, i16, i16, i32, [256 x %struct.mng_palette8e], i32, [256 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i32, ptr, i16, i16, i16, ptr, i8, i8, i32, i32, i32, i32, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i8, i8, i8, i32, ptr, ptr, i16, i16, i16, i16, i32, i32, ptr, %struct.z_stream, i32, i32, i32, i32, i32, i32, i8, i8, [256 x i32], i8 }
	%struct.mng_palette8e = type { i8, i8, i8 }
	%struct.mng_pushdata = type { ptr, ptr, i32, i8, ptr, i32 }
	%struct.mng_savedata = type { i8, i8, i8, i8, i8, i8, i8, i16, i16, i16, i8, i16, i8, i8, i32, i32, i8, i32, i32, i32, i32, i32, [256 x %struct.mng_palette8e], i32, [256 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i32, ptr, i16, i16, i16 }
	%struct.z_stream = type { ptr, i32, i32, ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, i32, i32, i32 }

define void @mng_write_basi(ptr %src1, ptr %src2) {
entry:
	%tmp = load i8, ptr %src1 ; <i8> [#uses=1]
	%tmp.upgrd.1 = icmp ugt i8 %tmp, 8		; <i1> [#uses=1]
	%tmp.upgrd.2 = load i16, ptr %src2; <i16> [#uses=2]
	%tmp3 = icmp eq i16 %tmp.upgrd.2, 255		; <i1> [#uses=1]
	%tmp7 = icmp eq i16 %tmp.upgrd.2, -1		; <i1> [#uses=1]
	%bOpaque.0.in = select i1 %tmp.upgrd.1, i1 %tmp7, i1 %tmp3		; <i1> [#uses=1]
	br i1 %bOpaque.0.in, label %cond_next90, label %bb95

cond_next90:		; preds = %entry
	ret void

bb95:		; preds = %entry
	ret void
}
