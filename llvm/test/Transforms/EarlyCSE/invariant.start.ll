; RUN: opt < %s -S -early-cse | FileCheck %s
; RUN: opt < %s -S -passes=early-cse | FileCheck %s

declare {}* @llvm.invariant.start.p0i8(i64, i8* nocapture) nounwind readonly
declare void @llvm.invariant.end.p0i8({}*, i64, i8* nocapture) nounwind

; Check that we do load-load forwarding over invariant.start, since it does not
; clobber memory
define i8 @test_bypass1(i8 *%P) {
  ; CHECK-LABEL: @test_bypass1(
  ; CHECK-NEXT: %V1 = load i8, i8* %P
  ; CHECK-NEXT: %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
  ; CHECK-NEXT: ret i8 0

  %V1 = load i8, i8* %P
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
  %V2 = load i8, i8* %P
  %Diff = sub i8 %V1, %V2
  ret i8 %Diff
}


; Trivial Store->load forwarding over invariant.start
define i8 @test_bypass2(i8 *%P) {
  ; CHECK-LABEL: @test_bypass2(
  ; CHECK-NEXT: store i8 42, i8* %P
  ; CHECK-NEXT: %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
  ; CHECK-NEXT: ret i8 42

  store i8 42, i8* %P
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
  %V1 = load i8, i8* %P
  ret i8 %V1
}

; We can DSE over invariant.start calls, since the first store to
; %P is valid, and the second store is actually unreachable based on semantics
; of invariant.start.
define void @test_bypass3(i8* %P) {
; CHECK-LABEL: @test_bypass3(
; CHECK-NEXT:  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
; CHECK-NEXT:  store i8 60, i8* %P

  store i8 50, i8* %P
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
  store i8 60, i8* %P
  ret void
}


; FIXME: Now the first store can actually be eliminated, since there is no read within
; the invariant region, between start and end.
define void @test_bypass4(i8* %P) {

; CHECK-LABEL: @test_bypass4(
; CHECK-NEXT: store i8 50, i8* %P
; CHECK-NEXT:  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
; CHECK-NEXT: call void @llvm.invariant.end.p0i8({}* %i, i64 1, i8* %P)
; CHECK-NEXT:  store i8 60, i8* %P


  store i8 50, i8* %P
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
  call void @llvm.invariant.end.p0i8({}* %i, i64 1, i8* %P)
  store i8 60, i8* %P
  ret void
}


declare void @clobber()
declare {}* @llvm.invariant.start.p0i32(i64 %size, i32* nocapture %ptr)
declare void @llvm.invariant.end.p0i32({}*, i64, i32* nocapture) nounwind

define i32 @test_before_load(i32* %p) {
; CHECK-LABEL: @test_before_load
; CHECK: ret i32 0
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  %v1 = load i32, i32* %p
  call void @clobber()
  %v2 = load i32, i32* %p
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

define i32 @test_before_clobber(i32* %p) {
; CHECK-LABEL: @test_before_clobber
; CHECK: ret i32 0
  %v1 = load i32, i32* %p
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  call void @clobber()
  %v2 = load i32, i32* %p
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

define i32 @test_duplicate_scope(i32* %p) {
; CHECK-LABEL: @test_duplicate_scope
; CHECK: ret i32 0
  %v1 = load i32, i32* %p
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  call void @clobber()
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  %v2 = load i32, i32* %p
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

define i32 @test_unanalzyable_load(i32* %p) {
; CHECK-LABEL: @test_unanalzyable_load
; CHECK: ret i32 0
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  call void @clobber()
  %v1 = load i32, i32* %p
  call void @clobber()
  %v2 = load i32, i32* %p
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

define i32 @test_negative_after_clobber(i32* %p) {
; CHECK-LABEL: @test_negative_after_clobber
; CHECK: ret i32 %sub
  %v1 = load i32, i32* %p
  call void @clobber()
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  %v2 = load i32, i32* %p
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

define i32 @test_merge(i32* %p, i1 %cnd) {
; CHECK-LABEL: @test_merge
; CHECK: ret i32 0
  %v1 = load i32, i32* %p
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  br i1 %cnd, label %merge, label %taken

taken:
  call void @clobber()
  br label %merge
merge:
  %v2 = load i32, i32* %p
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

define i32 @test_negative_after_mergeclobber(i32* %p, i1 %cnd) {
; CHECK-LABEL: @test_negative_after_mergeclobber
; CHECK: ret i32 %sub
  %v1 = load i32, i32* %p
  br i1 %cnd, label %merge, label %taken

taken:
  call void @clobber()
  br label %merge
merge:
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  %v2 = load i32, i32* %p
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

; In theory, this version could work, but earlycse is incapable of
; merging facts along distinct paths.  
define i32 @test_false_negative_merge(i32* %p, i1 %cnd) {
; CHECK-LABEL: @test_false_negative_merge
; CHECK: ret i32 %sub
  %v1 = load i32, i32* %p
  br i1 %cnd, label %merge, label %taken

taken:
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  call void @clobber()
  br label %merge
merge:
  %v2 = load i32, i32* %p
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

define i32 @test_merge_unanalyzable_load(i32* %p, i1 %cnd) {
; CHECK-LABEL: @test_merge_unanalyzable_load
; CHECK: ret i32 0
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  call void @clobber()
  %v1 = load i32, i32* %p
  br i1 %cnd, label %merge, label %taken

taken:
  call void @clobber()
  br label %merge
merge:
  %v2 = load i32, i32* %p
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

define void @test_dse_before_load(i32* %p, i1 %cnd) {
; CHECK-LABEL: @test_dse_before_load
; CHECK-NOT: store
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  %v1 = load i32, i32* %p
  call void @clobber()
  store i32 %v1, i32* %p
  ret void
}

define void @test_dse_after_load(i32* %p, i1 %cnd) {
; CHECK-LABEL: @test_dse_after_load
; CHECK-NOT: store
  %v1 = load i32, i32* %p
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  call void @clobber()
  store i32 %v1, i32* %p
  ret void
}


; In this case, we have a false negative since MemoryLocation is implicitly
; typed due to the user of a Value to represent the address.  Note that other
; passes will canonicalize away the bitcasts in this example.
define i32 @test_false_negative_types(i32* %p) {
; CHECK-LABEL: @test_false_negative_types
; CHECK: ret i32 %sub
  call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  %v1 = load i32, i32* %p
  call void @clobber()
  %pf = bitcast i32* %p to float*
  %v2f = load float, float* %pf
  %v2 = bitcast float %v2f to i32
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

define i32 @test_negative_size1(i32* %p) {
; CHECK-LABEL: @test_negative_size1
; CHECK: ret i32 %sub
  call {}* @llvm.invariant.start.p0i32(i64 3, i32* %p)
  %v1 = load i32, i32* %p
  call void @clobber()
  %v2 = load i32, i32* %p
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

define i32 @test_negative_size2(i32* %p) {
; CHECK-LABEL: @test_negative_size2
; CHECK: ret i32 %sub
  call {}* @llvm.invariant.start.p0i32(i64 0, i32* %p)
  %v1 = load i32, i32* %p
  call void @clobber()
  %v2 = load i32, i32* %p
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

define i32 @test_negative_scope(i32* %p) {
; CHECK-LABEL: @test_negative_scope
; CHECK: ret i32 %sub
  %scope = call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  call void @llvm.invariant.end.p0i32({}* %scope, i64 4, i32* %p)
  %v1 = load i32, i32* %p
  call void @clobber()
  %v2 = load i32, i32* %p
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

define i32 @test_false_negative_scope(i32* %p) {
; CHECK-LABEL: @test_false_negative_scope
; CHECK: ret i32 %sub
  %scope = call {}* @llvm.invariant.start.p0i32(i64 4, i32* %p)
  %v1 = load i32, i32* %p
  call void @clobber()
  %v2 = load i32, i32* %p
  call void @llvm.invariant.end.p0i32({}* %scope, i64 4, i32* %p)
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}

; Invariant load defact starts an invariant.start scope of the appropriate size
define i32 @test_invariant_load_scope(i32* %p) {
; CHECK-LABEL: @test_invariant_load_scope
; CHECK: ret i32 0
  %v1 = load i32, i32* %p, !invariant.load !{}
  call void @clobber()
  %v2 = load i32, i32* %p
  %sub = sub i32 %v1, %v2
  ret i32 %sub
}
