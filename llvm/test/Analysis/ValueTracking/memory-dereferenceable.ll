; RUN: opt -passes=print-memderefs -S < %s -disable-output  -use-dereferenceable-at-point-semantics=0 2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL
; RUN: opt -passes=print-memderefs -S < %s -disable-output  -use-dereferenceable-at-point-semantics=1 2>&1 | FileCheck %s --check-prefixes=CHECK,POINT


; Uses the print-deref (+ analyze to print) pass to run
; isDereferenceablePointer() on many load instruction operands

target datalayout = "e-i32:32:64"

%TypeOpaque = type opaque

declare zeroext i1 @return_i1()

declare ptr @foo()
@globalstr = global [6 x i8] c"hello\00"
@globali32ptr = external global ptr

%struct.A = type { [8 x i8], [5 x i8] }
@globalstruct = external global %struct.A

@globalptr.align1 = external global i8, align 1
@globalptr.align16 = external global i8, align 16

; Loads from sret arguments
; CHECK-LABEL: 'test_sret'
; CHECK: %sret_gep{{.*}}(aligned)
; CHECK-NOT: %sret_gep_outside
define void @test_sret(ptr sret(%struct.A) %result, i1 %arg) {
  %sret_gep = getelementptr inbounds %struct.A, ptr %result, i64 0, i32 1, i64 2
  load i8, ptr %sret_gep

  %sret_gep_outside = getelementptr %struct.A, ptr %result, i64 0, i32 1, i64 7
  load i8, ptr %sret_gep_outside
  ret void
}

; CHECK-LABEL: 'test'
define void @test(ptr addrspace(1) dereferenceable(8) %dparam,
                  ptr addrspace(1) dereferenceable(32) align 1 %dparam.align1,
                  ptr addrspace(1) dereferenceable(32) align 16 %dparam.align16)
    gc "statepoint-example" {
; CHECK: The following are dereferenceable:
entry:
  call void @mayfree()

; GLOBAL: %dparam{{.*}}(unaligned)
; POINT-NOT: %dparam{{.*}}(unaligned)
    %load3 = load i32, ptr addrspace(1) %dparam

; GLOBAL: %relocate{{.*}}(unaligned)
; POINT-NOT: %relocate{{.*}}(unaligned)
    %tok = tail call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live" (ptr addrspace(1) %dparam)]
    %relocate = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %tok, i32 0, i32 0)
    %load4 = load i32, ptr addrspace(1) %relocate

; CHECK-NOT: %nparam
    %dpa = call ptr addrspace(1) @func1(ptr addrspace(1) %dparam)
    %nparam = getelementptr i32, ptr addrspace(1) %dpa, i32 5
    %load5 = load i32, ptr addrspace(1) %nparam

    ; Load from a non-dereferenceable load
; CHECK-NOT: %nd_load
    %nd_load = load ptr, ptr @globali32ptr
    %load6 = load i32, ptr %nd_load

    ; Load from a dereferenceable load
; GLOBAL: %d4_load{{.*}}(unaligned)
; POINT-NOT: %d4_load{{.*}}(unaligned)
    %d4_load = load ptr, ptr @globali32ptr, !dereferenceable !0
    %load7 = load i32, ptr %d4_load

    ; Load from an offset not covered by the dereferenceable portion
; CHECK-NOT: %d2_load
    %d2_load = load ptr, ptr @globali32ptr, !dereferenceable !1
    %load8 = load i32, ptr %d2_load

    ; Load from a potentially null pointer with dereferenceable_or_null
; CHECK-NOT: %d_or_null_load
    %d_or_null_load = load ptr, ptr @globali32ptr, !dereferenceable_or_null !0
    %load9 = load i32, ptr %d_or_null_load

    ; Load from a non-null pointer with dereferenceable_or_null
; GLOBAL: %d_or_null_non_null_load{{.*}}(unaligned)
; POINT-NOT: %d_or_null_non_null_load{{.*}}(unaligned)
    %d_or_null_non_null_load = load ptr, ptr @globali32ptr, !nonnull !2, !dereferenceable_or_null !0
    %load10 = load i32, ptr %d_or_null_non_null_load

    ; Loads from aligned arguments
; GLOBAL: %dparam.align1{{.*}}(unaligned)
; POINT-NOT: %dparam.align1{{.*}}(unaligned)
; POINT-NOT: %dparam.align16{{.*}}(aligned)
; GLOBAL: %dparam.align16{{.*}}(aligned)
    %load15 = load i8, ptr addrspace(1) %dparam.align1, align 16
    %load16 = load i8, ptr addrspace(1) %dparam.align16, align 16

    ; Loads from GEPs
; GLOBAL: %gep.align1.offset1{{.*}}(unaligned)
; GLOBAL: %gep.align16.offset1{{.*}}(unaligned)
; GLOBAL: %gep.align1.offset16{{.*}}(unaligned)
; GLOBAL: %gep.align16.offset16{{.*}}(aligned)
; POINT-NOT: %gep.align1.offset1{{.*}}(unaligned)
; POINT-NOT: %gep.align16.offset1{{.*}}(unaligned)
; POINT-NOT: %gep.align1.offset16{{.*}}(unaligned)
; POINT-NOT: %gep.align16.offset16{{.*}}(aligned)
    %gep.align1.offset1 = getelementptr inbounds i8, ptr addrspace(1) %dparam.align1, i32 1
    %gep.align16.offset1 = getelementptr inbounds i8, ptr addrspace(1) %dparam.align16, i32 1
    %gep.align1.offset16 = getelementptr inbounds i8, ptr addrspace(1) %dparam.align1, i32 16
    %gep.align16.offset16 = getelementptr inbounds i8, ptr addrspace(1) %dparam.align16, i32 16
    %load19 = load i8, ptr addrspace(1) %gep.align1.offset1, align 16
    %load20 = load i8, ptr addrspace(1) %gep.align16.offset1, align 16
    %load21 = load i8, ptr addrspace(1) %gep.align1.offset16, align 16
    %load22 = load i8, ptr addrspace(1) %gep.align16.offset16, align 16

; CHECK-NOT: %no_deref_return
; GLOBAL: %deref_return{{.*}}(unaligned)
; GLOBAL: %deref_and_aligned_return{{.*}}(aligned)
; POINT-NOT: %deref_return{{.*}}(unaligned)
; POINT-NOT: %deref_and_aligned_return{{.*}}(aligned)
    %no_deref_return = call ptr @foo()
    %deref_return = call dereferenceable(32) ptr @foo()
    %deref_and_aligned_return = call dereferenceable(32) align 16 ptr @foo()
    %load23 = load i32, ptr %no_deref_return
    %load24 = load i32, ptr %deref_return, align 16
    %load25 = load i32, ptr %deref_and_aligned_return, align 16

    ; Load from a dereferenceable and aligned load
; GLOBAL: %d4_unaligned_load{{.*}}(unaligned)
; GLOBAL: %d4_aligned_load{{.*}}(aligned)
; POINT-NOT: %d4_unaligned_load{{.*}}(unaligned)
; POINT-NOT: %d4_aligned_load{{.*}}(aligned)
    %d4_unaligned_load = load ptr, ptr @globali32ptr, !dereferenceable !0
    %d4_aligned_load = load ptr, ptr @globali32ptr, !dereferenceable !0, !align !{i64 16}
    %load26 = load i32, ptr %d4_unaligned_load, align 16
    %load27 = load i32, ptr %d4_aligned_load, align 16
    ret void
}

; Loads from aligned allocas
; CHECK-LABEL: 'alloca_aligned'
; CHECK: %alloca.align1{{.*}}(unaligned)
; CHECK: %alloca.align16{{.*}}(aligned)
define void @alloca_aligned() {
   %alloca.align1 = alloca i1, align 1
   %alloca.align16 = alloca i1, align 16
   call void @mayfree()
   %load17 = load i1, ptr %alloca.align1, align 16
   %load18 = load i1, ptr %alloca.align16, align 16
   ret void
}

; CHECK-LABEL: 'alloca_basic'
; CHECK: %alloca{{.*}}(aligned)
define void @alloca_basic() {
  %alloca = alloca i1
  call void @mayfree()
  %load2 = load i1, ptr %alloca
  ret void
}

; Load from empty array alloca
; CHECK-LABEL: 'alloca_empty'
; CHECK-NOT: %empty_alloca
define void @alloca_empty() {
  %empty_alloca = alloca i8, i64 0
  call void @mayfree()
  %empty_load = load i8, ptr %empty_alloca
  ret void
}

; Alloca with no explicit alignment is aligned to preferred alignment of
; the type (specified by datalayout string).
; CHECK-LABEL: 'alloca_perfalign'
; CHECK: %alloca.noalign{{.*}}(aligned)
define void @alloca_perfalign() {
   %alloca.noalign = alloca i32
   call void @mayfree()
   %load28 = load i32, ptr %alloca.noalign, align 8
   ret void
}

; CHECK-LABEL: 'global'
; CHECK: @globalptr.align1{{.*}}(unaligned)
; CHECK: @globalptr.align16{{.*}}(aligned)
; CHECK: @globalstr{{.*}}(aligned)
define void @global() {
  %load13 = load i8, ptr @globalptr.align1, align 16
  %load14 = load i8, ptr @globalptr.align16, align 16

  %load1 = load i8, ptr @globalstr
  ret void
}

; It's OK to overrun static array size as long as we stay within underlying
; object size
; CHECK-LABEL: 'global_allocationsize'
; CHECK: %within_allocation{{.*}}(aligned)
; CHECK-NOT: %outside_allocation
define void @global_allocationsize() {
  %within_allocation = getelementptr inbounds %struct.A, ptr @globalstruct, i64 0, i32 0, i64 10
  %load11 = load i8, ptr %within_allocation

  %outside_allocation = getelementptr inbounds %struct.A, ptr @globalstruct, i64 0, i32 1, i64 10
  %load12 = load i8, ptr %outside_allocation
  ret void
}

; Loads from byval arguments
; CHECK-LABEL: 'byval'
; CHECK: %i8_byval{{.*}}(aligned)
; CHECK-NOT: %bad_byval_cast
; CHECK: %byval_gep{{.*}}(aligned)
; CHECK: %A_byval{{.*}}(unaligned)
define void @byval(ptr byval(i8) %i8_byval,
                   ptr byval(%struct.A) %A_byval) {
  call void @mayfree()
  load i8, ptr %i8_byval

  load i32, ptr %i8_byval

  %byval_gep = getelementptr inbounds %struct.A, ptr %A_byval, i64 0, i32 1, i64 2
  load i8, ptr %byval_gep
  load i32, ptr %A_byval

  ret void
}

; CHECK-LABEL: 'f_0'
; GLOBAL: %ptr = inttoptr i32 %val to ptr, !dereferenceable !0
; POINT-NOT: %ptr = inttoptr i32 %val to ptr, !dereferenceable !0
define i32 @f_0(i32 %val) {
  %ptr = inttoptr i32 %val to ptr, !dereferenceable !0
  call void @mayfree()
  %load29 = load i32, ptr %ptr, align 8
  ret i32 %load29
}


; The most basic case showing the difference between legacy global deref
; attribute semantics and the new point-in-time semantics.
; CHECK-LABEL: 'negative'
; GLOBAL: %p
; POINT-NOT: %p
define void @negative(ptr dereferenceable(8) %p) {
  call void @mayfree()
  %v = load i32, ptr %p
  ret void
}

; CHECK-LABEL: 'infer_func_attrs1'
; CHECK: %p
define void @infer_func_attrs1(ptr dereferenceable(8) %p) nofree nosync {
  call void @mayfree()
  %v = load i32, ptr %p
  ret void
}

; CHECK-LABEL: 'infer_func_attrs2'
; GLOBAL: %p
; POINT-NOT: %p
; FIXME: Can be inferred from attributes
define void @infer_func_attrs2(ptr dereferenceable(8) %p) readonly {
  call void @mayfree()
  %v = load i32, ptr %p
  ret void
}

; CHECK-LABEL: 'infer_noalias1'
; GLOBAL: %p
; POINT-NOT: %p
; FIXME: Can be inferred from attributes
define void @infer_noalias1(ptr dereferenceable(8) noalias nofree %p) {
  call void @mayfree()
  %v = load i32, ptr %p
  ret void
}

; CHECK-LABEL: 'infer_noalias2'
; GLOBAL: %p
; POINT-NOT: %p
; FIXME: Can be inferred from attributes
define void @infer_noalias2(ptr dereferenceable(8) noalias readonly %p) nosync {
  call void @mayfree()
  %v = load i32, ptr %p
  ret void
}


; Just check that we don't crash.
; CHECK-LABEL: 'opaque_type_crasher'
define void @opaque_type_crasher(ptr dereferenceable(16) %a, i1 %arg) {
entry:
  %ptr8 = getelementptr inbounds i8, ptr %a, i32 8
  br i1 %arg, label %if.then, label %if.end

if.then:
  %res = load i32, ptr %ptr8, align 4
  br label %if.end

if.end:
  ret void
}

declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token, i32, i32)

declare ptr addrspace(1) @func1(ptr addrspace(1) returned) nounwind argmemonly

; Can free any object accessible in memory
declare void @mayfree()

!0 = !{i64 4}
!1 = !{i64 2}
!2 = !{}
