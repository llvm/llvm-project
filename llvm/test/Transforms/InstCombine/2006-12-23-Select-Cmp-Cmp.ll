; For PR1065. This causes an assertion in instcombine if a select with two cmp
; operands is encountered.
; RUN: opt < %s -instcombine -disable-output
; END.

target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.internal_state = type { i32 }
	%struct.mng_data = type { i32, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i32, i32, i32, i8, i32, i32, i32, i32, i16, i16, i16, i8, i8, double, double, double, i8, i8, i8, i8, i32, i32, i32, i32, i32, i8, i32, i32, i8*, i8* (i32)*, void (i8*, i32)*, void (i8*, i8*, i32)*, i8 (%struct.mng_data*)*, i8 (%struct.mng_data*)*, i8 (%struct.mng_data*, i8*, i32, i32*)*, i8 (%struct.mng_data*, i8*, i32, i32*)*, i8 (%struct.mng_data*, i32, i8, i32, i32, i32, i32, i8*)*, i8 (%struct.mng_data*, i32, i32, i8*)*, i8 (%struct.mng_data*, i32, i32)*, i8 (%struct.mng_data*, i8, i8*, i8*, i8*, i8*)*, i8 (%struct.mng_data*)*, i8 (%struct.mng_data*, i8*)*, i8 (%struct.mng_data*, i8*)*, i8 (%struct.mng_data*, i32, i32)*, i8 (%struct.mng_data*, i32, i32, i8*)*, i8 (%struct.mng_data*, i8, i8, i32, i32)*, i8* (%struct.mng_data*, i32)*, i8* (%struct.mng_data*, i32)*, i8* (%struct.mng_data*, i32)*, i8 (%struct.mng_data*, i32, i32, i32, i32)*, i32 (%struct.mng_data*)*, i8 (%struct.mng_data*, i32)*, i8 (%struct.mng_data*, i32)*, i8 (%struct.mng_data*, i32, i32, i32, i32, i32, i32, i32, i32)*, i8 (%struct.mng_data*, i8)*, i8 (%struct.mng_data*, i32, i8*)*, i8 (%struct.mng_data*, i32, i8, i8*)*, i8, i32, i32, i8*, i8*, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i8, i8, i8, i8, i8, i32, i8, i8, i8, i32, i8*, i32, i8*, i32, i8, i8, i8, i32, i8*, i8*, i32, i32, i8*, i8*, %struct.mng_pushdata*, %struct.mng_pushdata*, %struct.mng_pushdata*, %struct.mng_pushdata*, i8, i8, i32, i32, i8*, i8, i8, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8, i32, i32, i8*, i32, i32, i32, i8, i8, i32, i32, i32, i32, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i8*, i8*, i8*, i32, i8*, i8*, i8*, i8*, i8*, %struct.mng_savedata*, i32, i32, i32, i32, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8*, i8*, i8, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8*, i8*, i8*, i8*, i8*, [256 x i8], double, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, i16, i8, i8, i8, i8, i8, i32, i32, i8, i32, i32, i32, i32, i16, i16, i16, i8, i16, i8, i32, i32, i32, i32, i8, i32, i32, i8, i32, i32, i32, i32, i8, i32, i32, i8, i32, i32, i32, i32, i32, i8, i32, i8, i16, i16, i16, i16, i32, [256 x %struct.mng_palette8e], i32, [256 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i32, i8*, i16, i16, i16, i8*, i8, i8, i32, i32, i32, i32, i8, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, i8*, i8, i8, i8, i32, i8*, i8*, i16, i16, i16, i16, i32, i32, i8*, %struct.z_stream, i32, i32, i32, i32, i32, i32, i8, i8, [256 x i32], i8 }
	%struct.mng_palette8e = type { i8, i8, i8 }
	%struct.mng_pushdata = type { i8*, i8*, i32, i8, i8*, i32 }
	%struct.mng_savedata = type { i8, i8, i8, i8, i8, i8, i8, i16, i16, i16, i8, i16, i8, i8, i32, i32, i8, i32, i32, i32, i32, i32, [256 x %struct.mng_palette8e], i32, [256 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i32, i8*, i16, i16, i16 }
	%struct.z_stream = type { i8*, i32, i32, i8*, i32, i32, i8*, %struct.internal_state*, i8* (i8*, i32, i32)*, void (i8*, i8*)*, i8*, i32, i32, i32 }

define void @mng_write_basi() {
entry:
	%tmp = load i8, i8* null		; <i8> [#uses=1]
	%tmp.upgrd.1 = icmp ugt i8 %tmp, 8		; <i1> [#uses=1]
	%tmp.upgrd.2 = load i16, i16* null		; <i16> [#uses=2]
	%tmp3 = icmp eq i16 %tmp.upgrd.2, 255		; <i1> [#uses=1]
	%tmp7 = icmp eq i16 %tmp.upgrd.2, -1		; <i1> [#uses=1]
	%bOpaque.0.in = select i1 %tmp.upgrd.1, i1 %tmp7, i1 %tmp3		; <i1> [#uses=1]
	br i1 %bOpaque.0.in, label %cond_next90, label %bb95

cond_next90:		; preds = %entry
	ret void

bb95:		; preds = %entry
	ret void
}
