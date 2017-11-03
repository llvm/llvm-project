; RUN: opt -S -rewrite-statepoints-for-gc < %s | FileCheck %s

; This test checks that metadata that's invalid after RS4GC is dropped. 
; We can miscompile if optimizations scheduled after RS4GC uses the
; metadata that's infact invalid.

declare void @bar()

declare void @baz(i32)
; Confirm that loadedval instruction does not contain invariant.load metadata.
; but contains the range metadata.
; Since loadedval is not marked invariant, it will prevent incorrectly sinking
; %loadedval in LICM and avoid creation of an unrelocated use of %baseaddr.
define void @test_invariant_load() gc "statepoint-example" {
; CHECK-LABEL: @test_invariant_load
; CHECK: %loadedval = load i32, i32 addrspace(1)* %baseaddr, align 8, !range !0
bb:
  br label %outerloopHdr

outerloopHdr:                                              ; preds = %bb6, %bb
  %baseaddr = phi i32 addrspace(1)* [ undef, %bb ], [ %tmp4, %bb6 ]
; LICM may sink this load to exit block after RS4GC because it's tagged invariant.
  %loadedval = load i32, i32 addrspace(1)* %baseaddr, align 8, !range !0, !invariant.load !1
  br label %innerloopHdr

innerloopHdr:                                              ; preds = %innerlooplatch, %outerloopHdr
  %tmp4 = phi i32 addrspace(1)* [ %baseaddr, %outerloopHdr ], [ %gep, %innerlooplatch ]
  br label %innermostloophdr

innermostloophdr:                                              ; preds = %bb6, %innerloopHdr
  br i1 undef, label %exitblock, label %bb6

bb6:                                              ; preds = %innermostloophdr
  switch i32 undef, label %innermostloophdr [
    i32 0, label %outerloopHdr
    i32 1, label %innerlooplatch
  ]

innerlooplatch:                                              ; preds = %bb6
  call void @bar()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %tmp4, i64 8
  br label %innerloopHdr

exitblock:                                             ; preds = %innermostloophdr
  %tmp13 = add i32 42, %loadedval
  call void @baz(i32 %tmp13)
  unreachable
}

; drop the noalias metadata.
define void @test_noalias(i32 %x, i32 addrspace(1)* %p, i32 addrspace(1)* %q) gc "statepoint-example" {
; CHECK-LABEL: test_noalias
; CHECK: %y = load i32, i32 addrspace(1)* %q, align 16
; CHECK: gc.statepoint
; CHECK: %p.relocated
; CHECK-NEXT: %p.relocated.casted = bitcast i8 addrspace(1)* %p.relocated to i32 addrspace(1)*
; CHECK-NEXT: store i32 %x, i32 addrspace(1)* %p.relocated.casted, align 16
entry:
  %y = load i32, i32 addrspace(1)* %q, align 16, !noalias !3
  call void @baz(i32 %x)
  store i32 %x, i32 addrspace(1)* %p, align 16, !noalias !4
  ret void
}

; drop the dereferenceable metadata
define void @test_dereferenceable(i32 addrspace(1)* addrspace(1)* %p, i32 %x, i32 addrspace(1)* %q) gc "statepoint-example" {
; CHECK-LABEL: test_dereferenceable
; CHECK: %v1 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* %p
; CHECK-NEXT: %v2 = load i32, i32 addrspace(1)* %v1
; CHECK: gc.statepoint
  %v1 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* %p, !dereferenceable !5
  %v2 = load i32, i32 addrspace(1)* %v1
  call void @baz(i32 %x)
  store i32 %v2, i32 addrspace(1)* %q, align 16
  ret void
}

; invariant.start allows us to sink the load past the baz statepoint call into taken block, which is
; incorrect. remove the invariant.start and RAUW undef.
define void @test_inv_start(i1 %cond, i32 addrspace(1)* addrspace(1)* %p, i32 %x, i32 addrspace(1)* %q) gc "statepoint-example" {
; CHECK-LABEL: test_inv_start
; CHECK-NOT: invariant.start
; CHECK: gc.statepoint
  %v1 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* %p
  %invst = call {}* @llvm.invariant.start.p1i32(i64 1, i32 addrspace(1)* %v1)
  %v2 = load i32, i32 addrspace(1)* %v1
  call void @baz(i32 %x)
  br i1 %cond, label %taken, label %untaken

taken:
  store i32 %v2, i32 addrspace(1)* %q, align 16
  call void @llvm.invariant.end.p1i32({}* %invst, i64 4, i32 addrspace(1)* %v1)
  ret void

; CHECK-LABEL: untaken:
; CHECK: gc.statepoint
untaken:
  %foo = call i32 @escaping.invariant.start({}* %invst)
  call void @dummy(i32 %foo)
  ret void
}

; invariant.start is removed and the uses are undef'ed.
define void @test_inv_start2(i1 %cond, i32 addrspace(1)* addrspace(1)* %p, i32 %x, i32 addrspace(1)* %q) gc "statepoint-example" {
; CHECK-LABEL: test_inv_start2
; CHECK-NOT: invariant.start
; CHECK: gc.statepoint
  %v1 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* %p
  %invst = call {}* @llvm.invariant.start.p1i32(i64 1, i32 addrspace(1)* %v1)
  %v2 = load i32, i32 addrspace(1)* %v1
  call void @baz(i32 %x)
  br i1 %cond, label %taken, label %untaken

taken:
  store i32 %v2, i32 addrspace(1)* %q, align 16
  call void @llvm.invariant.end.p1i32({}* %invst, i64 4, i32 addrspace(1)* %v1)
  ret void

untaken:
  ret void
}
declare {}* @llvm.invariant.start.p1i32(i64, i32 addrspace(1)*  nocapture) nounwind readonly
declare void @llvm.invariant.end.p1i32({}*, i64, i32 addrspace(1)* nocapture) nounwind
declare i32 @escaping.invariant.start({}*) nounwind
declare void @dummy(i32)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidi32f(i64, i32, void (i32)*, i32, i32, ...)

; Function Attrs: nounwind readonly
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32, i32) #0

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)

attributes #0 = { nounwind readonly }

!0 = !{i32 0, i32 2147483647}
!1 = !{}
!2 = !{i32 10, i32 1}
!3 = !{!3}
!4 = !{!4}
!5 = !{i64 8}
