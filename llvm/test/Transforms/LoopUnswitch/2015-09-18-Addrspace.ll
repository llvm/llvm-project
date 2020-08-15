; RUN: opt < %s -loop-unswitch -enable-new-pm=0 -S | FileCheck %s
; RUN: opt < %s -loop-unswitch -enable-new-pm=0 -enable-mssa-loop-dependency=true -verify-memoryssa -S | FileCheck %s

; In cases where two address spaces do not have the same size pointer, the
; input for the addrspacecast should not be used as a substitute for itself
; when manipulating the pointer.

target datalayout = "e-m:e-p:16:16-p1:32:16-i32:16-i64:16-n8:16"

define void @foo() {
; CHECK-LABEL: @foo
entry:
  %arrayidx.i1 = getelementptr inbounds i16, i16* undef, i16 undef
  %arrayidx.i = addrspacecast i16* %arrayidx.i1 to i16 addrspace(1)*
  br i1 undef, label %for.body.i, label %bar.exit

for.body.i:                                       ; preds = %for.body.i, %entry
; When we call makeLoopInvariant (i.e. trivial LICM) on this load, it 
; will try to find the base object to prove deferenceability.  If we look
; through the addrspacecast, we'll fail an assertion about bitwidths matching
; CHECK-LABEL: for.body.i
; CHECK:   %0 = load i16, i16 addrspace(1)* %arrayidx.i, align 2
  %0 = load i16, i16 addrspace(1)* %arrayidx.i, align 2
  %cmp1.i = icmp eq i16 %0, 0
  br i1 %cmp1.i, label %bar.exit, label %for.body.i

bar.exit:                                         ; preds = %for.body.i, %entry
  ret void
}
