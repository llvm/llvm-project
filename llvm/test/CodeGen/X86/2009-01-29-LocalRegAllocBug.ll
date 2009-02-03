; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin9.6 -regalloc=local -disable-fp-elim
; rdar://6538384

	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.Lit = type { i32 }
	%struct.StreamBuffer = type { %struct.FILE*, [1048576 x i8], i32, i32 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }

declare fastcc i32 @_Z8parseIntI12StreamBufferEiRT_(%struct.StreamBuffer*)

declare i8* @llvm.eh.exception() nounwind

define i32 @main(i32 %argc, i8** nocapture %argv) noreturn {
entry:
	%0 = invoke fastcc i32 @_Z8parseIntI12StreamBufferEiRT_(%struct.StreamBuffer* null)
			to label %bb1.i16.i.i unwind label %lpad.i.i		; <i32> [#uses=0]

bb1.i16.i.i:		; preds = %entry
	br i1 false, label %bb.i.i.i.i, label %_ZN3vecI3LitE4pushERKS0_.exit.i.i.i

bb.i.i.i.i:		; preds = %bb1.i16.i.i
	br label %_ZN3vecI3LitE4pushERKS0_.exit.i.i.i

_ZN3vecI3LitE4pushERKS0_.exit.i.i.i:		; preds = %bb.i.i.i.i, %bb1.i16.i.i
	%lits.i.i.0.0 = phi %struct.Lit* [ null, %bb1.i16.i.i ], [ null, %bb.i.i.i.i ]		; <%struct.Lit*> [#uses=1]
	%1 = invoke fastcc i32 @_Z8parseIntI12StreamBufferEiRT_(%struct.StreamBuffer* null)
			to label %.noexc21.i.i unwind label %lpad.i.i		; <i32> [#uses=0]

.noexc21.i.i:		; preds = %_ZN3vecI3LitE4pushERKS0_.exit.i.i.i
	unreachable

lpad.i.i:		; preds = %_ZN3vecI3LitE4pushERKS0_.exit.i.i.i, %entry
	%lits.i.i.0.3 = phi %struct.Lit* [ %lits.i.i.0.0, %_ZN3vecI3LitE4pushERKS0_.exit.i.i.i ], [ null, %entry ]		; <%struct.Lit*> [#uses=1]
	%eh_ptr.i.i = call i8* @llvm.eh.exception()		; <i8*> [#uses=0]
	free %struct.Lit* %lits.i.i.0.3
	unreachable
}
