; RUN: llc -mtriple=hexagon -mattr=+reserved-r19 < %s | FileCheck %s
;; Test that the backend fatally errors without reserved-r19 (backstop for
;; the driver diagnostic in SanitizerArgs.cpp).
; RUN: not --crash llc -mtriple=hexagon < %s 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: llc -mtriple=hexagon -mattr=+reserved-r19 < %s | FileCheck %s --check-prefix=CFI
; RUN: llc -mtriple=hexagon-unknown-linux-musl -mattr=+reserved-r19 < %s | FileCheck %s --check-prefix=MUSL

;; Leaf function - no LR spill, SCS should not emit any r19 instructions.
; CHECK-LABEL: leaf:
; CHECK-NOT: r19
; CHECK: jumpr r31

;; Non-leaf function - SCS emits prologue (addi + store) and epilogue (load + addi).
;; The SCS store is fused into the same packet as the first call; because
;; Hexagon packets use old-value reads the original R31 is saved regardless.
;; The epilogue load and addi are also in the same packet; the load uses the
;; old (pre-decrement) r19 value per Hexagon packet semantics, and the -4
;; offset correctly addresses the saved slot.
; CHECK-LABEL: nonleaf:
; CHECK:      r19 = add(r19,#4)
; CHECK:      call bar
; CHECK:      memw(r19+#-4) = r31
; CHECK:      {
; CHECK-DAG:  r19 = add(r19,#-4)
; CHECK-DAG:  r31 = memw(r19+#-4)
; CHECK:      }
; CHECK:      jumpr r31

;; Multi-call function - only one SCS prologue/epilogue pair, not one per call.
; CHECK-LABEL: twocalls:
; CHECK:      r19 = add(r19,#4)
; CHECK:      call bar
; CHECK:      memw(r19+#-4) = r31
; CHECK:      call bar
; CHECK:      {
; CHECK-DAG:  r19 = add(r19,#-4)
; CHECK-DAG:  r31 = memw(r19+#-4)
; CHECK:      }
; CHECK:      jumpr r31

;; Conditional call (shrink-wrapping): the early-return path is a leaf and
;; has no SCS prologue/epilogue.  The call path gets the SCS pair.
; CHECK-LABEL: condcall:
; CHECK:       if (!p0.new) jumpr:nt r31
; CHECK:       r19 = add(r19,#4)
; CHECK:       call bar
; CHECK:       memw(r19+#-4) = r31
; CHECK:       {
; CHECK-DAG:   r19 = add(r19,#-4)
; CHECK-DAG:   r31 = memw(r19+#-4)
; CHECK:       }
; CHECK:       jumpr r31

;; Tail call - SCS prologue and epilogue are both emitted; the epilogue
;; instructions and the tail jump are fused into the same packet.
; CHECK-LABEL: tailcall:
; CHECK:      r19 = add(r19,#4)
; CHECK:      memw(r19+#-4) = r31
; CHECK:      {
; CHECK-DAG:  r19 = add(r19,#-4)
; CHECK-DAG:  r31 = memw(r19+#-4)
; CHECK-DAG:  jump bar
; CHECK:      }

;; Noreturn call - SCS prologue is emitted but no SCS epilogue since the
;; function never returns.
; CHECK-LABEL: noret:
; CHECK:      r19 = add(r19,#4)
; CHECK:      memw(r19+#-4) = r31
; CHECK-NOT:  r31 = memw
; CHECK-NOT:  r19 = add(r19,#-4)
; CHECK-LABEL: nonleaf_cfi:

;; Minsize + multiple callee-saved registers: the restore stub
;; (__restore_r16_through_r17_and_deallocframe) must NOT be used when SCS is
;; active because it performs deallocframe+jumpr without the SCS epilogue.
; CHECK-LABEL: minsize_multicall:
; CHECK:      r19 = add(r19,#4)
; CHECK:      memw(r19+#-4) = r31
; CHECK:      {
; CHECK-DAG:  r19 = add(r19,#-4)
; CHECK-DAG:  r31 = memw(r19+#-4)
; CHECK:      }
; CHECK-NOT:  __restore_
; CHECK:      jumpr r31

;; Minsize + tail call + multiple callee-saved registers: the tailcall restore
;; stub (__restore_r16_through_r17_and_deallocframe_before_tailcall) must NOT be
;; used when SCS is active.  The SCS epilogue and tail jump are fused together.
; CHECK-LABEL: minsize_tailcall:
; CHECK:      r19 = add(r19,#4)
; CHECK:      memw(r19+#-4) = r31
; CHECK:      {
; CHECK-DAG:  r19 = add(r19,#-4)
; CHECK-DAG:  r31 = memw(r19+#-4)
; CHECK-DAG:  jump bar
; CHECK:      }
; CHECK-NOT:  __restore_

;; Multiple return paths - each exit block gets its own SCS epilogue.
; CHECK-LABEL: multi_return:
; CHECK:      r19 = add(r19,#4)
; CHECK:      memw(r19+#-4) = r31
; CHECK:      r31 = memw(r19+#-4)
; CHECK:      r19 = add(r19,#-4)
; CHECK:      jumpr r31
; CHECK:      r31 = memw(r19+#-4)
; CHECK:      r19 = add(r19,#-4)
; CHECK:      jumpr r31

;; Without r19 reserved, SCS should report an error.
; ERR: Must reserve r19 to use shadow call stack on Hexagon

;; Non-leaf with uwtable - exercises CFI escape (DW_CFA_val_expression for r19)
;; and cfi_restore on epilogue.
; CFI-LABEL: nonleaf_cfi:
; CFI:        r19 = add(r19,#4)
; CFI:        memw(r19+#-4) = r31
; CFI:        .cfi_escape 0x16, 0x13, 0x02, 0x83, 0x7c
; CFI:        {
; CFI-DAG:    r31 = memw(r19+#-4)
; CFI-DAG:    r19 = add(r19,#-4)
; CFI:        }
; CFI:        .cfi_restore r19
; CFI:        jumpr r31

;; Musl vararg - exercises the vararg epilogue path with SCS.
; MUSL-LABEL: vararg_musl:
; MUSL:       r19 = add(r19,#4)
; MUSL:       memw(r19+#-4) = r31
; MUSL:       {
; MUSL-DAG:   r19 = add(r19,#-4)
; MUSL-DAG:   r31 = memw(r19+#-4)
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
