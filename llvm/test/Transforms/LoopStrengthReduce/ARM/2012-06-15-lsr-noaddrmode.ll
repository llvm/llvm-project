; RUN: llc < %s -O3 -march=thumb -mcpu=cortex-a8 | FileCheck %s
;
; LSR should only check for valid address modes when the IV user is a
; memory address.
; svn r158536, rdar://11635990
;
; Note that we still don't produce the best code here because we fail
; to coalesce the IV. See <rdar://problem/11680670> [coalescer] IVs
; need to be scheduled to expose coalescing.

; LSR before the fix:
;The chosen solution requires 4 regs, with addrec cost 1, plus 3 base adds, plus 2 setup cost:
;  LSR Use: Kind=Special, Offsets={0}, all-fixups-outside-loop, widest fixup type: i32
;    reg(%v3) + reg({0,+,-1}<%while.cond.i.i>) + imm(1)
;  LSR Use: Kind=ICmpZero, Offsets={0}, widest fixup type: i32
;    reg(%v3) + reg({0,+,-1}<%while.cond.i.i>)
;  LSR Use: Kind=Address of i32, Offsets={0}, widest fixup type: i32*
;    reg((-4 + (4 * %v3) + %v1)) + 4*reg({0,+,-1}<%while.cond.i.i>)
;  LSR Use: Kind=Address of i32, Offsets={0}, widest fixup type: i32*
;    reg((-4 + (4 * %v3) + %v4)) + 4*reg({0,+,-1}<%while.cond.i.i>)
;  LSR Use: Kind=Special, Offsets={0}, all-fixups-outside-loop, widest fixup type: i32
;    reg(%v3)
;
; LSR after the fix:
;The chosen solution requires 4 regs, with addrec cost 1, plus 1 base add, plus 2 setup cost:
;  LSR Use: Kind=Special, Offsets={0}, all-fixups-outside-loop, widest fixup type: i32
;    reg({%v3,+,-1}<nsw><%while.cond.i.i>) + imm(1)
;  LSR Use: Kind=ICmpZero, Offsets={0}, widest fixup type: i32
;    reg({%v3,+,-1}<nsw><%while.cond.i.i>)
;  LSR Use: Kind=Address of i32, Offsets={0}, widest fixup type: i32*
;    reg((-4 + %v1)) + 4*reg({%v3,+,-1}<nsw><%while.cond.i.i>)
;  LSR Use: Kind=Address of i32, Offsets={0}, widest fixup type: i32*
;    reg((-4 + %v4)) + 4*reg({%v3,+,-1}<nsw><%while.cond.i.i>)
;  LSR Use: Kind=Special, Offsets={0}, all-fixups-outside-loop, widest fixup type: i32
;    reg(%v3)


%s = type { i32* }

@ncol = external global i32, align 4

declare i32* @getptr() nounwind
declare %s* @getstruct() nounwind

; CHECK: @main
; Check that the loop preheader contains no address computation.
; CHECK: %end_of_chain
; CHECK-NOT: add{{.*}}lsl
; CHECK: ldr{{.*}}lsl #2
; CHECK: ldr{{.*}}lsl #2
define i32 @main() nounwind ssp {
entry:
  %v0 = load i32* @ncol, align 4, !tbaa !0
  %v1 = tail call i32* @getptr() nounwind
  %cmp10.i = icmp eq i32 %v0, 0
  br label %while.cond.outer

while.cond.outer:
  %call18 = tail call %s* @getstruct() nounwind
  br label %while.cond

while.cond:
  %cmp20 = icmp eq i32* %v1, null
  br label %while.body

while.body:
  %v3 = load i32* @ncol, align 4, !tbaa !0
  br label %end_of_chain

end_of_chain:
  %state.i = getelementptr inbounds %s* %call18, i32 0, i32 0
  %v4 = load i32** %state.i, align 4, !tbaa !3
  br label %while.cond.i.i

while.cond.i.i:
  %counter.0.i.i = phi i32 [ %v3, %end_of_chain ], [ %dec.i.i, %land.rhs.i.i ]
  %dec.i.i = add nsw i32 %counter.0.i.i, -1
  %tobool.i.i = icmp eq i32 %counter.0.i.i, 0
  br i1 %tobool.i.i, label %where.exit, label %land.rhs.i.i

land.rhs.i.i:
  %arrayidx.i.i = getelementptr inbounds i32* %v4, i32 %dec.i.i
  %v5 = load i32* %arrayidx.i.i, align 4, !tbaa !0
  %arrayidx1.i.i = getelementptr inbounds i32* %v1, i32 %dec.i.i
  %v6 = load i32* %arrayidx1.i.i, align 4, !tbaa !0
  %cmp.i.i = icmp eq i32 %v5, %v6
  br i1 %cmp.i.i, label %while.cond.i.i, label %equal_data.exit.i

equal_data.exit.i:
  ret i32 %counter.0.i.i

where.exit:
  br label %while.end.i

while.end.i:
  ret i32 %v3
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
!3 = metadata !{metadata !"any pointer", metadata !1}
