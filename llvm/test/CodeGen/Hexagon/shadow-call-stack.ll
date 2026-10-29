;; r18 is the default shadow call stack pointer register; reserving it is all
;; that is required.
; RUN: llc -mtriple=hexagon -mattr=+reserved-r18 < %s | FileCheck %s
; RUN: llc -mtriple=hexagon -mattr=+reserved-r18 < %s | FileCheck %s --check-prefix=CFI
; RUN: llc -mtriple=hexagon-unknown-linux-musl -mattr=+reserved-r18 < %s | FileCheck %s --check-prefix=MUSL

;; The backend fatally errors unless the SCS register is reserved (backstop for
;; the driver diagnostic in SanitizerArgs.cpp).  Reserving some other register
;; does not help.
; RUN: not --crash llc -mtriple=hexagon < %s 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not --crash llc -mtriple=hexagon -mattr=+reserved-r19 < %s 2>&1 | FileCheck %s --check-prefix=ERR

;; scs-reg-rN selects a different register; the diagnostic follows it.
; RUN: llc -mtriple=hexagon -mattr=+scs-reg-r16,+reserved-r16 < %s \
; RUN:   | FileCheck %s --check-prefix=R16
; RUN: not --crash llc -mtriple=hexagon -mattr=+scs-reg-r16,+reserved-r18 < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ERR16
; RUN: not --crash llc -mtriple=hexagon -mattr=+scs-reg-r16,+scs-reg-r17 < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ERRMULTI

;; Only one scs-reg-rN may be given.
; ERRMULTI: Only one shadow call stack pointer register may be selected

;; With r16 selected, the prologue/epilogue use r16 and leave r18 alone.
; R16-LABEL: nonleaf:
; R16:      r16 = add(r16,#4)
; R16:      call bar
; R16:      memw(r16+#-4) = r31
; R16:      {
; R16-DAG:  r16 = add(r16,#-4)
; R16-DAG:  r31 = memw(r16+#-4)
; R16:      }
; R16:      jumpr r31

;; Every spill stub saves the range starting at r16, so with r16 as the SCS
;; register no stub can ever be used - reserving r16 removes d8 from the
;; callee-saved set entirely.
; R16-LABEL: minsize_multicall:
; R16-NOT:  __save_
; R16-NOT:  __restore_

;; Leaf function - no LR spill, SCS should not emit any r18 instructions.
; CHECK-LABEL: leaf:
; CHECK-NOT: r18
; CHECK: jumpr r31

;; Non-leaf function - SCS emits prologue (addi + store) and epilogue (load + addi).
;; The SCS store is fused into the same packet as the first call; because
;; Hexagon packets use old-value reads the original R31 is saved regardless.
;; The epilogue load and addi are also in the same packet; the load uses the
;; old (pre-decrement) r18 value per Hexagon packet semantics, and the -4
;; offset correctly addresses the saved slot.
; CHECK-LABEL: nonleaf:
; CHECK:      r18 = add(r18,#4)
; CHECK:      call bar
; CHECK:      memw(r18+#-4) = r31
; CHECK:      {
; CHECK-DAG:  r18 = add(r18,#-4)
; CHECK-DAG:  r31 = memw(r18+#-4)
; CHECK:      }
; CHECK:      jumpr r31

;; Multi-call function - only one SCS prologue/epilogue pair, not one per call.
; CHECK-LABEL: twocalls:
; CHECK:      r18 = add(r18,#4)
; CHECK:      call bar
; CHECK:      memw(r18+#-4) = r31
; CHECK:      call bar
; CHECK:      {
; CHECK-DAG:  r18 = add(r18,#-4)
; CHECK-DAG:  r31 = memw(r18+#-4)
; CHECK:      }
; CHECK:      jumpr r31

;; Conditional call (shrink-wrapping): the early-return path is a leaf and
;; has no SCS prologue/epilogue.  The call path gets the SCS pair.
; CHECK-LABEL: condcall:
; CHECK:       if (!p0.new) jumpr:nt r31
; CHECK:       r18 = add(r18,#4)
; CHECK:       call bar
; CHECK:       memw(r18+#-4) = r31
; CHECK:       {
; CHECK-DAG:   r18 = add(r18,#-4)
; CHECK-DAG:   r31 = memw(r18+#-4)
; CHECK:       }
; CHECK:       jumpr r31

;; Tail call - SCS prologue and epilogue are both emitted; the epilogue
;; instructions and the tail jump are fused into the same packet.
; CHECK-LABEL: tailcall:
; CHECK:      r18 = add(r18,#4)
; CHECK:      memw(r18+#-4) = r31
; CHECK:      {
; CHECK-DAG:  r18 = add(r18,#-4)
; CHECK-DAG:  r31 = memw(r18+#-4)
; CHECK-DAG:  jump bar
; CHECK:      }

;; Noreturn call - SCS prologue is emitted but no SCS epilogue since the
;; function never returns.
; CHECK-LABEL: noret:
; CHECK:      r18 = add(r18,#4)
; CHECK:      memw(r18+#-4) = r31
; CHECK-NOT:  r31 = memw
; CHECK-NOT:  r18 = add(r18,#-4)
; CHECK-LABEL: nonleaf_cfi:

;; Minsize + multiple callee-saved registers: the restore stub
;; (__restore_r16_through_r17_and_deallocframe) must NOT be used when SCS is
;; active because it performs deallocframe+jumpr without the SCS epilogue.
; CHECK-LABEL: minsize_multicall:
; CHECK:      r18 = add(r18,#4)
; CHECK:      memw(r18+#-4) = r31
; CHECK:      {
; CHECK-DAG:  r18 = add(r18,#-4)
; CHECK-DAG:  r31 = memw(r18+#-4)
; CHECK:      }
; CHECK-NOT:  __restore_
; CHECK:      jumpr r31

;; Minsize + tail call + multiple callee-saved registers: the tailcall restore
;; stub (__restore_r16_through_r17_and_deallocframe_before_tailcall) must NOT be
;; used when SCS is active.  The SCS epilogue and tail jump are fused together.
; CHECK-LABEL: minsize_tailcall:
; CHECK:      r18 = add(r18,#4)
; CHECK:      memw(r18+#-4) = r31
; CHECK:      {
; CHECK-DAG:  r18 = add(r18,#-4)
; CHECK-DAG:  r31 = memw(r18+#-4)
; CHECK-DAG:  jump bar
; CHECK:      }
; CHECK-NOT:  __restore_

;; Multiple return paths - each exit block gets its own SCS epilogue.
; CHECK-LABEL: multi_return:
; CHECK:      r18 = add(r18,#4)
; CHECK:      memw(r18+#-4) = r31
; CHECK:      r31 = memw(r18+#-4)
; CHECK:      r18 = add(r18,#-4)
; CHECK:      jumpr r31
; CHECK:      r31 = memw(r18+#-4)
; CHECK:      r18 = add(r18,#-4)
; CHECK:      jumpr r31

;; A minsize function using many callee-saved registers - without SCS this is
;; the shape that gets a spill/restore stub.  Reserving the SCS register breaks
;; the r19:18 double, which leaves r19 in the callee-saved set as a lone single
;; register and forces inline spills, so no stub covering r18 can be selected.
;; This locks in the invariant that a stub never reaches the SCS register.
; CHECK-LABEL: minsize_manycsr:
; CHECK-NOT:  __save_
; CHECK-NOT:  __restore_
; CHECK:      jumpr r31

;; Without the SCS register reserved, SCS should report an error naming it.
; ERR: Must reserve r18 to use shadow call stack on Hexagon
; ERR16: Must reserve r16 to use shadow call stack on Hexagon

;; Non-leaf with uwtable - exercises CFI escape (DW_CFA_val_expression for r18)
;; and cfi_restore on epilogue.
; CFI-LABEL: nonleaf_cfi:
; CFI:        r18 = add(r18,#4)
; CFI:        memw(r18+#-4) = r31
; CFI:        .cfi_escape 0x16, 0x12, 0x02, 0x82, 0x7c
; CFI:        {
; CFI-DAG:    r31 = memw(r18+#-4)
; CFI-DAG:    r18 = add(r18,#-4)
; CFI:        }
; CFI:        .cfi_restore r18
; CFI:        jumpr r31

;; Musl vararg - exercises the vararg epilogue path with SCS.
; MUSL-LABEL: vararg_musl:
; MUSL:       r18 = add(r18,#4)
; MUSL:       memw(r18+#-4) = r31
; MUSL:       {
; MUSL-DAG:   r18 = add(r18,#-4)
; MUSL-DAG:   r31 = memw(r18+#-4)
; MUSL:       }
; MUSL:       jumpr r31

declare i32 @foo(i32)
declare void @bar()
declare void @baz(i32)

define void @leaf() shadowcallstack nounwind {
  ret void
}

define void @nonleaf() shadowcallstack nounwind {
  call void @bar()
  ret void
}

define void @twocalls() shadowcallstack nounwind {
  call void @bar()
  call void @bar()
  ret void
}

define void @condcall(i1 %cond) shadowcallstack nounwind {
  br i1 %cond, label %call, label %ret
call:
  call void @bar()
  br label %ret
ret:
  ret void
}

define void @tailcall() shadowcallstack nounwind {
  call void @bar()
  tail call void @bar()
  ret void
}

define void @noret() shadowcallstack nounwind {
  call void @bar() noreturn
  unreachable
}

define void @nonleaf_cfi() shadowcallstack uwtable {
  call void @bar()
  ret void
}

define void @vararg_musl(i32 %a, ...) shadowcallstack nounwind {
  call void @bar()
  ret void
}

define i32 @minsize_multicall(i32 %x) shadowcallstack nounwind minsize
                              "disable-tail-calls"="true" {
  %call = call i32 @foo(i32 %x)
  %call1 = call i32 @foo(i32 %x)
  %sum = add i32 %call, %call1
  ret i32 %sum
}

define void @minsize_tailcall(i32 %x) shadowcallstack nounwind minsize {
  call void @baz(i32 %x)
  call void @baz(i32 %x)
  tail call void @bar()
  ret void
}

define i32 @multi_return(i32 %x) shadowcallstack nounwind optnone noinline {
entry:
  %call = call i32 @foo(i32 %x)
  %cmp = icmp sgt i32 %call, 0
  br i1 %cmp, label %pos, label %neg

pos:
  %r1 = call i32 @foo(i32 %call)
  ret i32 %r1

neg:
  %r2 = call i32 @foo(i32 0)
  ret i32 %r2
}

define i32 @minsize_manycsr(i32 %x) shadowcallstack nounwind minsize
                            "disable-tail-calls"="true" {
  %a = call i32 @foo(i32 %x)
  %b = call i32 @foo(i32 %a)
  %c = call i32 @foo(i32 %b)
  %d = call i32 @foo(i32 %c)
  %e = call i32 @foo(i32 %d)
  %f = call i32 @foo(i32 %e)
  %g = call i32 @foo(i32 %f)
  %s1 = add i32 %a, %b
  %s2 = add i32 %s1, %c
  %s3 = add i32 %s2, %d
  %s4 = add i32 %s3, %e
  %s5 = add i32 %s4, %f
  %s6 = add i32 %s5, %g
  ret i32 %s6
}
