; RUN: llc -O3 -mtriple=thumb-eabi -mcpu=cortex-a8 %s -o - -arm-atomic-cfg-tidy=0 | FileCheck %s
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


%s = type { ptr }

@ncol = external global i32, align 4

declare ptr @getptr() nounwind
declare ptr @getstruct() nounwind

; CHECK: @main
; Check that the loop preheader contains no address computation.
; CHECK: %while.cond.i.i
; CHECK-NOT: add{{.*}}lsl
; CHECK: ldr{{.*}}lsl #2
; CHECK: ldr{{.*}}lsl #2
define i32 @main() nounwind ssp {
entry:
  %v0 = load i32, ptr @ncol, align 4
  %v1 = tail call ptr @getptr() nounwind
  %cmp10.i = icmp eq i32 %v0, 0
  br label %while.cond.outer

while.cond.outer:
  %call18 = tail call ptr @getstruct() nounwind
  br label %while.cond

while.cond:
  %cmp20 = icmp eq ptr %v1, null
  br label %while.body

while.body:
  %v3 = load i32, ptr @ncol, align 4
  br label %end_of_chain

end_of_chain:
  %v4 = load ptr, ptr %call18, align 4
  br label %while.cond.i.i

while.cond.i.i:
  %counter.0.i.i = phi i32 [ %v3, %end_of_chain ], [ %dec.i.i, %land.rhs.i.i ]
  %dec.i.i = add nsw i32 %counter.0.i.i, -1
  %tobool.i.i = icmp eq i32 %counter.0.i.i, 0
  br i1 %tobool.i.i, label %where.exit, label %land.rhs.i.i

land.rhs.i.i:
  %arrayidx.i.i = getelementptr inbounds i32, ptr %v4, i32 %dec.i.i
  %v5 = load i32, ptr %arrayidx.i.i, align 4
  %arrayidx1.i.i = getelementptr inbounds i32, ptr %v1, i32 %dec.i.i
  %v6 = load i32, ptr %arrayidx1.i.i, align 4
  %cmp.i.i = icmp eq i32 %v5, %v6
  br i1 %cmp.i.i, label %while.cond.i.i, label %equal_data.exit.i

equal_data.exit.i:
  ret i32 %counter.0.i.i

where.exit:
  br label %while.end.i

while.end.i:
  ret i32 %v3
}
