; RUN: llc -verify-machineinstrs < %s

target triple = "powerpc-unknown-linux-gnu"
	%struct..0anon = type { i32 }
	%struct.A = type { %struct.anon }
	%struct.anon = type <{  }>

define void @bork(ptr %In0P) {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%i.035.0 = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=2]
	%tmp8 = getelementptr float, ptr %In0P, i32 %i.035.0		; <ptr> [#uses=2]
	%tmp21 = tail call i32 asm "lwbrx $0, $2, $1", "=r,r,bO,*m"(ptr %tmp8, i32 0, ptr elementtype(i32) %tmp8 )		; <i32> [#uses=0]
	%indvar.next = add i32 %i.035.0, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, 4		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb
	ret void
}
