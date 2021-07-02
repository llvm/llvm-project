; RUN: llc -O0 -stop-before=irtranslator -global-isel %s -o - | FileCheck %s
; RUN: llc -O0 -stop-after=irtranslator -verify-machineinstrs -global-isel %s -o - | FileCheck --check-prefixes CHECK,CHECK-MIR %s

; Check that when using GlobalISel, the StackProtector pass currently inserts
; both prologue and epilogue instrumentation because GlobalISel does not have
; the same epilogue insertion/optimization as SelectionDAG.

target triple = "aarch64-none-unknown-eabi"

define void @foo() ssp {
; CHECK-LABEL: entry:
; CHECK-NEXT:   %StackGuardSlot = alloca i8*
; CHECK-NEXT:   %0 = call i8* @llvm.stackguard()
; CHECK-NEXT:   call void @llvm.stackprotector(i8* %0, i8** %StackGuardSlot)
; CHECK-NEXT:   %buf = alloca [8 x i8], align 1
; CHECK-NEXT:   %1 = call i8* @llvm.stackguard()
; CHECK-NEXT:   %2 = load volatile i8*, i8** %StackGuardSlot
; CHECK-NEXT:   %3 = icmp eq i8* %1, %2
; CHECK-NEXT:   br i1 %3, label %SP_return, label %CallStackCheckFailBlk, !prof !0
;
; CHECK: SP_return:
; CHECK-NEXT:   ret void
;
; CHECK: CallStackCheckFailBlk:
; CHECK-NEXT:   call void @__stack_chk_fail()
; CHECK-NEXT:   unreachable

; CHECK-MIR: bb.1.entry:
; CHECK-MIR:   %0:_(p0) = G_FRAME_INDEX %stack.0.StackGuardSlot
; CHECK-MIR-NEXT:   %1:gpr64sp(p0) = LOAD_STACK_GUARD :: (dereferenceable invariant load (p0)  from @__stack_chk_guard)
; CHECK-MIR-NEXT:   %2:gpr64sp(p0) = LOAD_STACK_GUARD :: (dereferenceable invariant load (p0)  from @__stack_chk_guard)
; CHECK-MIR-NEXT:   G_STORE %2(p0), %0(p0) :: (volatile store (p0) into %stack.0.StackGuardSlot)
; CHECK-MIR-NEXT:   %3:_(p0) = G_FRAME_INDEX %stack.1.buf
; CHECK-MIR-NEXT:   %4:gpr64sp(p0) = LOAD_STACK_GUARD :: (dereferenceable invariant load (p0)  from @__stack_chk_guard)
; CHECK-MIR-NEXT:   %5:_(p0) = G_LOAD %0(p0) :: (volatile dereferenceable load (p0)  from %ir.StackGuardSlot)
; CHECK-MIR-NEXT:   %6:_(s1) = G_ICMP intpred(eq), %4(p0), %5
; CHECK-MIR-NEXT:   G_BRCOND %6(s1), %bb.2
; CHECK-MIR-NEXT:   G_BR %bb.3
;
; CHECK-MIR: bb.2.SP_return:
; CHECK-MIR-NEXT:   RET_ReallyLR
;
; CHECK-MIR: bb.3.CallStackCheckFailBlk:
; CHECK-MIR-NEXT:   ADJCALLSTACKDOWN 0, 0, implicit-def $sp, implicit $sp
; CHECK-MIR-NEXT:   BL @__stack_chk_fail, csr_aarch64_aapcs, implicit-def $lr, implicit $sp
; CHECK-MIR-NEXT:   ADJCALLSTACKUP 0, 0, implicit-def $sp, implicit $sp
entry:
  %buf = alloca [8 x i8], align 1
  ret void
}
