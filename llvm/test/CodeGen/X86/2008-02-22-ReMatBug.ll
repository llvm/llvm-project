; RUN: llvm-as < %s | llc -march=x86 -stats |& grep {Number of re-materialization} | grep 4
; RUN: llvm-as < %s | llc -march=x86 -stats |& grep {Number of dead spill slots removed}
; rdar://5761454

	%struct.quad_struct = type { i32, i32, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct* }

define  %struct.quad_struct* @MakeTree(i32 %size, i32 %center_x, i32 %center_y, i32 %lo_proc, i32 %hi_proc, %struct.quad_struct* %parent, i32 %ct, i32 %level) nounwind  {
entry:
	br i1 true, label %bb43.i, label %bb.i

bb.i:		; preds = %entry
	ret %struct.quad_struct* null

bb43.i:		; preds = %entry
	br i1 true, label %CheckOutside.exit40.i, label %bb11.i38.i

bb11.i38.i:		; preds = %bb43.i
	ret %struct.quad_struct* null

CheckOutside.exit40.i:		; preds = %bb43.i
	br i1 true, label %CheckOutside.exit30.i, label %bb11.i28.i

bb11.i28.i:		; preds = %CheckOutside.exit40.i
	ret %struct.quad_struct* null

CheckOutside.exit30.i:		; preds = %CheckOutside.exit40.i
	br i1 true, label %CheckOutside.exit20.i, label %bb11.i18.i

bb11.i18.i:		; preds = %CheckOutside.exit30.i
	ret %struct.quad_struct* null

CheckOutside.exit20.i:		; preds = %CheckOutside.exit30.i
	br i1 true, label %bb34, label %bb11.i8.i

bb11.i8.i:		; preds = %CheckOutside.exit20.i
	ret %struct.quad_struct* null

bb34:		; preds = %CheckOutside.exit20.i
	%tmp15.reg2mem.0 = sdiv i32 %size, 2		; <i32> [#uses=7]
	%tmp85 = sub i32 %center_y, %tmp15.reg2mem.0		; <i32> [#uses=2]
	%tmp88 = sub i32 %center_x, %tmp15.reg2mem.0		; <i32> [#uses=2]
	%tmp92 = tail call  %struct.quad_struct* @MakeTree( i32 %tmp15.reg2mem.0, i32 %tmp88, i32 %tmp85, i32 0, i32 %hi_proc, %struct.quad_struct* null, i32 2, i32 0 ) nounwind 		; <%struct.quad_struct*> [#uses=0]
	%tmp99 = add i32 0, %hi_proc		; <i32> [#uses=1]
	%tmp100 = sdiv i32 %tmp99, 2		; <i32> [#uses=1]
	%tmp110 = tail call  %struct.quad_struct* @MakeTree( i32 %tmp15.reg2mem.0, i32 0, i32 %tmp85, i32 0, i32 %tmp100, %struct.quad_struct* null, i32 3, i32 0 ) nounwind 		; <%struct.quad_struct*> [#uses=0]
	%tmp122 = add i32 %tmp15.reg2mem.0, %center_y		; <i32> [#uses=2]
	%tmp129 = tail call  %struct.quad_struct* @MakeTree( i32 %tmp15.reg2mem.0, i32 0, i32 %tmp122, i32 0, i32 0, %struct.quad_struct* null, i32 1, i32 0 ) nounwind 		; <%struct.quad_struct*> [#uses=0]
	%tmp147 = tail call  %struct.quad_struct* @MakeTree( i32 %tmp15.reg2mem.0, i32 %tmp88, i32 %tmp122, i32 %lo_proc, i32 0, %struct.quad_struct* null, i32 0, i32 0 ) nounwind 		; <%struct.quad_struct*> [#uses=0]
	unreachable
}
