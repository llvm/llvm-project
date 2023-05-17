; RUN: opt -passes=objc-arc -S < %s | FileCheck %s

%struct.__objcFastEnumerationState = type { i64, ptr, ptr, [5 x i64] }

@"\01L_OBJC_METH_VAR_NAME_" = internal global [43 x i8] c"countByEnumeratingWithState:objects:count:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_SELECTOR_REFERENCES_" = internal global ptr @"\01L_OBJC_METH_VAR_NAME_", section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@g = common global ptr null, align 8
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__DATA, __objc_imageinfo, regular, no_dead_strip"

declare void @callee()
declare ptr @returner()
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare ptr @llvm.objc.retain(ptr)
declare void @llvm.objc.enumerationMutation(ptr)
declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind
declare ptr @objc_msgSend(ptr, ptr, ...) nonlazybind
declare void @use(ptr)
declare void @llvm.objc.release(ptr)
declare ptr @def()
declare void @__crasher_block_invoke(ptr nocapture)
declare ptr @llvm.objc.retainBlock(ptr)
declare void @__crasher_block_invoke1(ptr nocapture)

!0 = !{}

; Delete a nested retain+release pair.

; CHECK-LABEL: define void @test0(
; CHECK: call ptr @llvm.objc.retain
; CHECK-NOT: @llvm.objc.retain
; CHECK: }
define void @test0(ptr %a) nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x ptr], align 8
  %0 = call ptr @llvm.objc.retain(ptr %a) nounwind
  call void @llvm.memset.p0.i64(ptr align 8 %state.ptr, i8 0, i64 64, i1 false)
  %1 = call ptr @llvm.objc.retain(ptr %0) nounwind
  %tmp2 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call = call i64 @objc_msgSend(ptr %1, ptr %tmp2, ptr %state.ptr, ptr %items.ptr, i64 16)
  %iszero = icmp eq i64 %call, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 2
  %mutationsptr = load ptr, ptr %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64, ptr %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call, %forcoll.loopinit ], [ %call6, %forcoll.refetch ]
  %tmp7 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp7, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.notmutated ]
  %mutationsptr3 = load ptr, ptr %mutationsptr.ptr, align 8
  %statemutations = load i64, ptr %mutationsptr3, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @llvm.objc.enumerationMutation(ptr %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load ptr, ptr %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr ptr, ptr %stateitems, i64 %forcoll.index
  %3 = load ptr, ptr %currentitem.ptr, align 8
  call void @use(ptr %3)
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp5 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call6 = call i64 @objc_msgSend(ptr %1, ptr %tmp5, ptr %state.ptr, ptr %items.ptr, i64 16)
  %5 = icmp eq i64 %call6, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @llvm.objc.release(ptr %1) nounwind
  call void @llvm.objc.release(ptr %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Delete a nested retain+release pair.

; CHECK-LABEL: define void @test2(
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue
; CHECK-NOT: @llvm.objc.retain
; CHECK: }
define void @test2() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x ptr], align 8
  %call = call ptr @returner()
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) nounwind
  call void @llvm.memset.p0.i64(ptr align 8 %state.ptr, i8 0, i64 64, i1 false)
  %1 = call ptr @llvm.objc.retain(ptr %0) nounwind
  %tmp2 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call3 = call i64 @objc_msgSend(ptr %1, ptr %tmp2, ptr %state.ptr, ptr %items.ptr, i64 16)
  %iszero = icmp eq i64 %call3, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 2
  %mutationsptr = load ptr, ptr %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64, ptr %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call3, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp8 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp8, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.notmutated ]
  %mutationsptr4 = load ptr, ptr %mutationsptr.ptr, align 8
  %statemutations = load i64, ptr %mutationsptr4, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @llvm.objc.enumerationMutation(ptr %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load ptr, ptr %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr ptr, ptr %stateitems, i64 %forcoll.index
  %3 = load ptr, ptr %currentitem.ptr, align 8
  call void @use(ptr %3)
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 @objc_msgSend(ptr %1, ptr %tmp6, ptr %state.ptr, ptr %items.ptr, i64 16)
  %5 = icmp eq i64 %call7, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @llvm.objc.release(ptr %1) nounwind
  call void @llvm.objc.release(ptr %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Delete a nested retain+release pair.

; CHECK-LABEL: define void @test4(
; CHECK: call ptr @llvm.objc.retain
; CHECK-NOT: @llvm.objc.retain
; CHECK: }
define void @test4() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x ptr], align 8
  %tmp = load ptr, ptr @g, align 8
  %0 = call ptr @llvm.objc.retain(ptr %tmp) nounwind
  call void @llvm.memset.p0.i64(ptr align 8 %state.ptr, i8 0, i64 64, i1 false)
  %1 = call ptr @llvm.objc.retain(ptr %0) nounwind
  %tmp4 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call = call i64 @objc_msgSend(ptr %1, ptr %tmp4, ptr %state.ptr, ptr %items.ptr, i64 16)
  %iszero = icmp eq i64 %call, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 2
  %mutationsptr = load ptr, ptr %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64, ptr %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call, %forcoll.loopinit ], [ %call8, %forcoll.refetch ]
  %tmp9 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp9, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.notmutated ]
  %mutationsptr5 = load ptr, ptr %mutationsptr.ptr, align 8
  %statemutations = load i64, ptr %mutationsptr5, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @llvm.objc.enumerationMutation(ptr %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load ptr, ptr %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr ptr, ptr %stateitems, i64 %forcoll.index
  %3 = load ptr, ptr %currentitem.ptr, align 8
  call void @use(ptr %3)
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp7 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call8 = call i64 @objc_msgSend(ptr %1, ptr %tmp7, ptr %state.ptr, ptr %items.ptr, i64 16)
  %5 = icmp eq i64 %call8, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @llvm.objc.release(ptr %1) nounwind
  call void @llvm.objc.release(ptr %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Delete a nested retain+release pair.

; CHECK-LABEL: define void @test5(
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue
; CHECK-NOT: @llvm.objc.retain
; CHECK: }
define void @test5() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x ptr], align 8
  %call = call ptr @returner()
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) nounwind
  call void @llvm.memset.p0.i64(ptr align 8 %state.ptr, i8 0, i64 64, i1 false)
  %1 = call ptr @llvm.objc.retain(ptr %0) nounwind
  %tmp2 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call3 = call i64 @objc_msgSend(ptr %1, ptr %tmp2, ptr %state.ptr, ptr %items.ptr, i64 16)
  %iszero = icmp eq i64 %call3, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 2
  %mutationsptr = load ptr, ptr %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64, ptr %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call3, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp8 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp8, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.notmutated ]
  %mutationsptr4 = load ptr, ptr %mutationsptr.ptr, align 8
  %statemutations = load i64, ptr %mutationsptr4, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @llvm.objc.enumerationMutation(ptr %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load ptr, ptr %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr ptr, ptr %stateitems, i64 %forcoll.index
  %3 = load ptr, ptr %currentitem.ptr, align 8
  call void @use(ptr %3)
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 @objc_msgSend(ptr %1, ptr %tmp6, ptr %state.ptr, ptr %items.ptr, i64 16)
  %5 = icmp eq i64 %call7, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @llvm.objc.release(ptr %1) nounwind
  call void @llvm.objc.release(ptr %0) nounwind, !clang.imprecise_release !0
  ret void
}

; We handle this now due to the fact that a release just needs a post dominating
; use.
;
; CHECK-LABEL: define void @test6(
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue
; CHECK-NOT: @llvm.objc.retain
; CHECK: }
define void @test6() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x ptr], align 8
  %call = call ptr @returner()
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) nounwind
  call void @llvm.memset.p0.i64(ptr align 8 %state.ptr, i8 0, i64 64, i1 false)
  %1 = call ptr @llvm.objc.retain(ptr %0) nounwind
  %tmp2 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call3 = call i64 @objc_msgSend(ptr %1, ptr %tmp2, ptr %state.ptr, ptr %items.ptr, i64 16)
  %iszero = icmp eq i64 %call3, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 2
  %mutationsptr = load ptr, ptr %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64, ptr %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call3, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp8 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp8, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.notmutated ]
  %mutationsptr4 = load ptr, ptr %mutationsptr.ptr, align 8
  %statemutations = load i64, ptr %mutationsptr4, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @llvm.objc.enumerationMutation(ptr %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load ptr, ptr %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr ptr, ptr %stateitems, i64 %forcoll.index
  %3 = load ptr, ptr %currentitem.ptr, align 8
  call void @use(ptr %3)
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 @objc_msgSend(ptr %1, ptr %tmp6, ptr %state.ptr, ptr %items.ptr, i64 16)
  %5 = icmp eq i64 %call7, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @llvm.objc.release(ptr %1) nounwind
  call void @callee()
  call void @llvm.objc.release(ptr %0) nounwind, !clang.imprecise_release !0
  ret void
}

; TODO: Delete a nested retain+release pair.
; The optimizer currently can't do this, because isn't isn't sophisticated enough in
; reasnoning about nesting.

; CHECK-LABEL: define void @test7(
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue
; CHECK: @llvm.objc.retain
; CHECK: }
define void @test7() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x ptr], align 8
  %call = call ptr @returner()
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) nounwind
  call void @callee()
  call void @llvm.memset.p0.i64(ptr align 8 %state.ptr, i8 0, i64 64, i1 false)
  %1 = call ptr @llvm.objc.retain(ptr %0) nounwind
  %tmp2 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call3 = call i64 @objc_msgSend(ptr %1, ptr %tmp2, ptr %state.ptr, ptr %items.ptr, i64 16)
  %iszero = icmp eq i64 %call3, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 2
  %mutationsptr = load ptr, ptr %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64, ptr %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call3, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp8 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp8, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.notmutated ]
  %mutationsptr4 = load ptr, ptr %mutationsptr.ptr, align 8
  %statemutations = load i64, ptr %mutationsptr4, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @llvm.objc.enumerationMutation(ptr %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load ptr, ptr %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr ptr, ptr %stateitems, i64 %forcoll.index
  %3 = load ptr, ptr %currentitem.ptr, align 8
  call void @use(ptr %3)
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 @objc_msgSend(ptr %1, ptr %tmp6, ptr %state.ptr, ptr %items.ptr, i64 16)
  %5 = icmp eq i64 %call7, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @llvm.objc.release(ptr %1) nounwind
  call void @callee()
  call void @llvm.objc.release(ptr %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Delete a nested retain+release pair.

; CHECK-LABEL: define void @test8(
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue
; CHECK-NOT: @llvm.objc.retain
; CHECK: }
define void @test8() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x ptr], align 8
  %call = call ptr @returner()
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) nounwind
  call void @llvm.memset.p0.i64(ptr align 8 %state.ptr, i8 0, i64 64, i1 false)
  %1 = call ptr @llvm.objc.retain(ptr %0) nounwind
  %tmp2 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call3 = call i64 @objc_msgSend(ptr %1, ptr %tmp2, ptr %state.ptr, ptr %items.ptr, i64 16)
  %iszero = icmp eq i64 %call3, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 2
  %mutationsptr = load ptr, ptr %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64, ptr %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call3, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp8 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp8, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.next ]
  %mutationsptr4 = load ptr, ptr %mutationsptr.ptr, align 8
  %statemutations = load i64, ptr %mutationsptr4, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @llvm.objc.enumerationMutation(ptr %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load ptr, ptr %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr ptr, ptr %stateitems, i64 %forcoll.index
  %3 = load ptr, ptr %currentitem.ptr, align 8
  %tobool = icmp eq ptr %3, null
  br i1 %tobool, label %forcoll.next, label %if.then

if.then:
  call void @callee()
  br label %forcoll.next

forcoll.next:
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 @objc_msgSend(ptr %1, ptr %tmp6, ptr %state.ptr, ptr %items.ptr, i64 16)
  %5 = icmp eq i64 %call7, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @llvm.objc.release(ptr %1) nounwind
  call void @llvm.objc.release(ptr %0) nounwind, !clang.imprecise_release !0
  ret void
}

; TODO: Delete a nested retain+release pair.
; The optimizer currently can't do this, because of a split loop backedge.
; See test9b for the same testcase without a split backedge.

; CHECK-LABEL: define void @test9(
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue
; CHECK: call ptr @llvm.objc.retain
; CHECK: }
define void @test9() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x ptr], align 8
  %call = call ptr @returner()
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) nounwind
  %call1 = call ptr @returner()
  %1 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call1) nounwind
  call void @llvm.memset.p0.i64(ptr align 8 %state.ptr, i8 0, i64 64, i1 false)
  %2 = call ptr @llvm.objc.retain(ptr %0) nounwind
  %tmp3 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call4 = call i64 @objc_msgSend(ptr %2, ptr %tmp3, ptr %state.ptr, ptr %items.ptr, i64 16)
  %iszero = icmp eq i64 %call4, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 2
  %mutationsptr = load ptr, ptr %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64, ptr %mutationsptr, align 8
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call4, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp9 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp9, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ %phitmp, %forcoll.notmutated.forcoll.loopbody_crit_edge ], [ 1, %forcoll.loopbody.outer ]
  %mutationsptr5 = load ptr, ptr %mutationsptr.ptr, align 8
  %statemutations = load i64, ptr %mutationsptr5, align 8
  %3 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %3, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @llvm.objc.enumerationMutation(ptr %2)
  br label %forcoll.notmutated

forcoll.notmutated:
  %exitcond = icmp eq i64 %forcoll.index, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.notmutated.forcoll.loopbody_crit_edge

forcoll.notmutated.forcoll.loopbody_crit_edge:
  %phitmp = add i64 %forcoll.index, 1
  br label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 @objc_msgSend(ptr %2, ptr %tmp6, ptr %state.ptr, ptr %items.ptr, i64 16)
  %4 = icmp eq i64 %call7, 0
  br i1 %4, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @llvm.objc.release(ptr %2) nounwind
  call void @llvm.objc.release(ptr %1) nounwind, !clang.imprecise_release !0
  call void @llvm.objc.release(ptr %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Like test9, but without a split backedge. TODO: optimize this.

; CHECK-LABEL: define void @test9b(
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue
; CHECK: @llvm.objc.retain
; CHECK: }
define void @test9b() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x ptr], align 8
  %call = call ptr @returner()
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) nounwind
  %call1 = call ptr @returner()
  %1 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call1) nounwind
  call void @llvm.memset.p0.i64(ptr align 8 %state.ptr, i8 0, i64 64, i1 false)
  %2 = call ptr @llvm.objc.retain(ptr %0) nounwind
  %tmp3 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call4 = call i64 @objc_msgSend(ptr %2, ptr %tmp3, ptr %state.ptr, ptr %items.ptr, i64 16)
  %iszero = icmp eq i64 %call4, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 2
  %mutationsptr = load ptr, ptr %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64, ptr %mutationsptr, align 8
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call4, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp9 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp9, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ %phitmp, %forcoll.notmutated ], [ 0, %forcoll.loopbody.outer ]
  %mutationsptr5 = load ptr, ptr %mutationsptr.ptr, align 8
  %statemutations = load i64, ptr %mutationsptr5, align 8
  %3 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %3, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @llvm.objc.enumerationMutation(ptr %2)
  br label %forcoll.notmutated

forcoll.notmutated:
  %phitmp = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %phitmp, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 @objc_msgSend(ptr %2, ptr %tmp6, ptr %state.ptr, ptr %items.ptr, i64 16)
  %4 = icmp eq i64 %call7, 0
  br i1 %4, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @llvm.objc.release(ptr %2) nounwind
  call void @llvm.objc.release(ptr %1) nounwind, !clang.imprecise_release !0
  call void @llvm.objc.release(ptr %0) nounwind, !clang.imprecise_release !0
  ret void
}

; TODO: Delete a nested retain+release pair.
; The optimizer currently can't do this, because of a split loop backedge.
; See test10b for the same testcase without a split backedge.

; CHECK-LABEL: define void @test10(
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue
; CHECK: call ptr @llvm.objc.retain
; CHECK: }
define void @test10() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x ptr], align 8
  %call = call ptr @returner()
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) nounwind
  %call1 = call ptr @returner()
  %1 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call1) nounwind
  call void @callee()
  call void @llvm.memset.p0.i64(ptr align 8 %state.ptr, i8 0, i64 64, i1 false)
  %2 = call ptr @llvm.objc.retain(ptr %0) nounwind
  %tmp3 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call4 = call i64 @objc_msgSend(ptr %2, ptr %tmp3, ptr %state.ptr, ptr %items.ptr, i64 16)
  %iszero = icmp eq i64 %call4, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 2
  %mutationsptr = load ptr, ptr %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64, ptr %mutationsptr, align 8
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call4, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp9 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp9, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ %phitmp, %forcoll.notmutated.forcoll.loopbody_crit_edge ], [ 1, %forcoll.loopbody.outer ]
  %mutationsptr5 = load ptr, ptr %mutationsptr.ptr, align 8
  %statemutations = load i64, ptr %mutationsptr5, align 8
  %3 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %3, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @llvm.objc.enumerationMutation(ptr %2)
  br label %forcoll.notmutated

forcoll.notmutated:
  %exitcond = icmp eq i64 %forcoll.index, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.notmutated.forcoll.loopbody_crit_edge

forcoll.notmutated.forcoll.loopbody_crit_edge:
  %phitmp = add i64 %forcoll.index, 1
  br label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 @objc_msgSend(ptr %2, ptr %tmp6, ptr %state.ptr, ptr %items.ptr, i64 16)
  %4 = icmp eq i64 %call7, 0
  br i1 %4, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @llvm.objc.release(ptr %2) nounwind
  call void @llvm.objc.release(ptr %1) nounwind, !clang.imprecise_release !0
  call void @llvm.objc.release(ptr %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Like test10, but without a split backedge. TODO: optimize this.

; CHECK-LABEL: define void @test10b(
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue
; CHECK: @llvm.objc.retain
; CHECK: }
define void @test10b() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x ptr], align 8
  %call = call ptr @returner()
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) nounwind
  %call1 = call ptr @returner()
  %1 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call1) nounwind
  call void @callee()
  call void @llvm.memset.p0.i64(ptr align 8 %state.ptr, i8 0, i64 64, i1 false)
  %2 = call ptr @llvm.objc.retain(ptr %0) nounwind
  %tmp3 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call4 = call i64 @objc_msgSend(ptr %2, ptr %tmp3, ptr %state.ptr, ptr %items.ptr, i64 16)
  %iszero = icmp eq i64 %call4, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState, ptr %state.ptr, i64 0, i32 2
  %mutationsptr = load ptr, ptr %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64, ptr %mutationsptr, align 8
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call4, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp9 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp9, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ %phitmp, %forcoll.notmutated ], [ 0, %forcoll.loopbody.outer ]
  %mutationsptr5 = load ptr, ptr %mutationsptr.ptr, align 8
  %statemutations = load i64, ptr %mutationsptr5, align 8
  %3 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %3, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @llvm.objc.enumerationMutation(ptr %2)
  br label %forcoll.notmutated

forcoll.notmutated:
  %phitmp = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %phitmp, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 @objc_msgSend(ptr %2, ptr %tmp6, ptr %state.ptr, ptr %items.ptr, i64 16)
  %4 = icmp eq i64 %call7, 0
  br i1 %4, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @llvm.objc.release(ptr %2) nounwind
  call void @llvm.objc.release(ptr %1) nounwind, !clang.imprecise_release !0
  call void @llvm.objc.release(ptr %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Pointers to strong pointers can obscure provenance relationships. Be conservative
; in the face of escaping pointers. rdar://12150909.

%struct.__block_d = type { i64, i64 }

@_NSConcreteStackBlock = external global ptr
@__block_d_tmp = external hidden constant { i64, i64, ptr, ptr, ptr, ptr }
@__block_d_tmp5 = external hidden constant { i64, i64, ptr, ptr, ptr, ptr }

; CHECK-LABEL: define void @test11(
; CHECK: tail call ptr @llvm.objc.retain(ptr %call) [[NUW:#[0-9]+]]
; CHECK: tail call ptr @llvm.objc.retain(ptr %call) [[NUW]]
; CHECK: call void @llvm.objc.release(ptr %call) [[NUW]], !clang.imprecise_release !0
; CHECK: }
define void @test11() {
entry:
  %block = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
  %block9 = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
  %call = call ptr @def(), !clang.arc.no_objc_arc_exceptions !0
  %foo = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 5
  %block.isa = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 0
  store ptr @_NSConcreteStackBlock, ptr %block.isa, align 8
  %block.flags = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 1
  store i32 1107296256, ptr %block.flags, align 8
  %block.reserved = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 2
  store i32 0, ptr %block.reserved, align 4
  %block.invoke = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 3
  store ptr @__crasher_block_invoke, ptr %block.invoke, align 8
  %block.d = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 4
  store ptr @__block_d_tmp, ptr %block.d, align 8
  %foo2 = tail call ptr @llvm.objc.retain(ptr %call) nounwind
  store ptr %foo2, ptr %foo, align 8
  %foo5 = call ptr @llvm.objc.retainBlock(ptr %block) nounwind
  call void @use(ptr %foo5), !clang.arc.no_objc_arc_exceptions !0
  call void @llvm.objc.release(ptr %foo5) nounwind
  %strongdestroy = load ptr, ptr %foo, align 8
  call void @llvm.objc.release(ptr %strongdestroy) nounwind, !clang.imprecise_release !0
  %foo10 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block9, i64 0, i32 5
  %block.isa11 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block9, i64 0, i32 0
  store ptr @_NSConcreteStackBlock, ptr %block.isa11, align 8
  %block.flags12 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block9, i64 0, i32 1
  store i32 1107296256, ptr %block.flags12, align 8
  %block.reserved13 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block9, i64 0, i32 2
  store i32 0, ptr %block.reserved13, align 4
  %block.invoke14 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block9, i64 0, i32 3
  store ptr @__crasher_block_invoke1, ptr %block.invoke14, align 8
  %block.d15 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block9, i64 0, i32 4
  store ptr @__block_d_tmp5, ptr %block.d15, align 8
  %foo18 = call ptr @llvm.objc.retain(ptr %call) nounwind
  store ptr %call, ptr %foo10, align 8
  %foo21 = call ptr @llvm.objc.retainBlock(ptr %block9) nounwind
  call void @use(ptr %foo21), !clang.arc.no_objc_arc_exceptions !0
  call void @llvm.objc.release(ptr %foo21) nounwind
  %strongdestroy25 = load ptr, ptr %foo10, align 8
  call void @llvm.objc.release(ptr %strongdestroy25) nounwind, !clang.imprecise_release !0
  call void @llvm.objc.release(ptr %call) nounwind, !clang.imprecise_release !0
  ret void
}


; CHECK: attributes [[NUW]] = { nounwind }
; CHECK: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: write) }
; CHECK: attributes #2 = { nonlazybind }
