; RUN: opt -print-memderefs -analyze -S <%s | FileCheck %s

; Uses the print-deref (+ analyze to print) pass to run
; isDereferenceablePointer() on many load instruction operands

target datalayout = "e"

declare zeroext i1 @return_i1()

declare i32* @foo()
@globalstr = global [6 x i8] c"hello\00"
@globali32ptr = external global i32*

%struct.A = type { [8 x i8], [5 x i8] }
@globalstruct = external global %struct.A

@globalptr.align1 = external global i8, align 1
@globalptr.align16 = external global i8, align 16

define void @test(i32 addrspace(1)* dereferenceable(8) %dparam,
                  i8 addrspace(1)* dereferenceable(32) align 1 %dparam.align1,
                  i8 addrspace(1)* dereferenceable(32) align 16 %dparam.align16)
    gc "statepoint-example" {
; CHECK: The following are dereferenceable:
entry:
; CHECK: %globalptr{{.*}}(aligned)
    %globalptr = getelementptr inbounds [6 x i8], [6 x i8]* @globalstr, i32 0, i32 0
    %load1 = load i8, i8* %globalptr

; CHECK: %alloca{{.*}}(aligned)
    %alloca = alloca i1
    %load2 = load i1, i1* %alloca

; CHECK: %dparam{{.*}}(aligned)
    %load3 = load i32, i32 addrspace(1)* %dparam

; CHECK: %relocate{{.*}}(aligned)
    %tok = tail call i32 (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, i32 addrspace(1)* %dparam)
    %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %tok, i32 7, i32 7)
    %load4 = load i32, i32 addrspace(1)* %relocate

; CHECK-NOT: %nparam
    %nparam = getelementptr i32, i32 addrspace(1)* %dparam, i32 5
    %load5 = load i32, i32 addrspace(1)* %nparam

    ; Load from a non-dereferenceable load
; CHECK-NOT: %nd_load
    %nd_load = load i32*, i32** @globali32ptr
    %load6 = load i32, i32* %nd_load

    ; Load from a dereferenceable load
; CHECK: %d4_load{{.*}}(aligned)
    %d4_load = load i32*, i32** @globali32ptr, !dereferenceable !0
    %load7 = load i32, i32* %d4_load

    ; Load from an offset not covered by the dereferenceable portion
; CHECK-NOT: %d2_load
    %d2_load = load i32*, i32** @globali32ptr, !dereferenceable !1
    %load8 = load i32, i32* %d2_load

    ; Load from a potentially null pointer with dereferenceable_or_null
; CHECK-NOT: %d_or_null_load
    %d_or_null_load = load i32*, i32** @globali32ptr, !dereferenceable_or_null !0
    %load9 = load i32, i32* %d_or_null_load

    ; Load from a non-null pointer with dereferenceable_or_null
; CHECK: %d_or_null_non_null_load{{.*}}(aligned)
    %d_or_null_non_null_load = load i32*, i32** @globali32ptr, !nonnull !2, !dereferenceable_or_null !0
    %load10 = load i32, i32* %d_or_null_non_null_load

    ; It's OK to overrun static array size as long as we stay within underlying object size
; CHECK: %within_allocation{{.*}}(aligned)
    %within_allocation = getelementptr inbounds %struct.A, %struct.A* @globalstruct, i64 0, i32 0, i64 10
    %load11 = load i8, i8* %within_allocation

    ; GEP is outside the underlying object size
; CHECK-NOT: %outside_allocation
    %outside_allocation = getelementptr inbounds %struct.A, %struct.A* @globalstruct, i64 0, i32 1, i64 10
    %load12 = load i8, i8* %outside_allocation

    ; Loads from aligned globals
; CHECK: @globalptr.align1{{.*}}(unaligned)
; CHECK: @globalptr.align16{{.*}}(aligned)
    %load13 = load i8, i8* @globalptr.align1, align 16
    %load14 = load i8, i8* @globalptr.align16, align 16

    ; Loads from aligned arguments
; CHECK: %dparam.align1{{.*}}(unaligned)
; CHECK: %dparam.align16{{.*}}(aligned)
    %load15 = load i8, i8 addrspace(1)* %dparam.align1, align 16
    %load16 = load i8, i8 addrspace(1)* %dparam.align16, align 16

    ; Loads from aligned allocas
; CHECK: %alloca.align1{{.*}}(unaligned)
; CHECK: %alloca.align16{{.*}}(aligned)
    %alloca.align1 = alloca i1, align 1
    %alloca.align16 = alloca i1, align 16
    %load17 = load i1, i1* %alloca.align1, align 16
    %load18 = load i1, i1* %alloca.align16, align 16

    ; Loads from GEPs
; CHECK: %gep.align1.offset1{{.*}}(unaligned)
; CHECK: %gep.align16.offset1{{.*}}(unaligned)
; CHECK: %gep.align1.offset16{{.*}}(unaligned)
; CHECK: %gep.align16.offset16{{.*}}(aligned)
    %gep.align1.offset1 = getelementptr inbounds i8, i8 addrspace(1)* %dparam.align1, i32 1
    %gep.align16.offset1 = getelementptr inbounds i8, i8 addrspace(1)* %dparam.align16, i32 1
    %gep.align1.offset16 = getelementptr inbounds i8, i8 addrspace(1)* %dparam.align1, i32 16
    %gep.align16.offset16 = getelementptr inbounds i8, i8 addrspace(1)* %dparam.align16, i32 16
    %load19 = load i8, i8 addrspace(1)* %gep.align1.offset1, align 16
    %load20 = load i8, i8 addrspace(1)* %gep.align16.offset1, align 16
    %load21 = load i8, i8 addrspace(1)* %gep.align1.offset16, align 16
    %load22 = load i8, i8 addrspace(1)* %gep.align16.offset16, align 16

; CHECK-NOT: %no_deref_return
; CHECK: %deref_return{{.*}}(unaligned)
; CHECK: %deref_and_aligned_return{{.*}}(aligned)
    %no_deref_return = call i32* @foo()
    %deref_return = call dereferenceable(32) i32* @foo()
    %deref_and_aligned_return = call dereferenceable(32) align 16 i32* @foo()
    %load23 = load i32, i32* %no_deref_return
    %load24 = load i32, i32* %deref_return, align 16
    %load25 = load i32, i32* %deref_and_aligned_return, align 16

    ; Load from a dereferenceable and aligned load
; CHECK: %d4_unaligned_load{{.*}}(unaligned)
; CHECK: %d4_aligned_load{{.*}}(aligned)
    %d4_unaligned_load = load i32*, i32** @globali32ptr, !dereferenceable !0
    %d4_aligned_load = load i32*, i32** @globali32ptr, !dereferenceable !0, !align !{i64 16}
    %load26 = load i32, i32* %d4_unaligned_load, align 16
    %load27 = load i32, i32* %d4_aligned_load, align 16

    ret void
}

declare i32 @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32, i32, i32)

!0 = !{i64 4}
!1 = !{i64 2}
!2 = !{}
