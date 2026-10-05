; RUN: opt -mtriple=x86_64-unknown-unknown -passes=x86-fence-nontemporal-stores -S < %s | FileCheck %s
; RUN: opt -mtriple=i386-unknown-unknown -mattr=-sse -passes=x86-fence-nontemporal-stores -S < %s | FileCheck %s --check-prefix=CHECK-NOSSE

declare void @sync_call()
declare void @nosync_call() nosync
declare void @nosync_nounwind_call() nosync nounwind
declare void @llvm.x86.sse.sfence()
declare void @llvm.x86.sse2.mfence()
declare i32 @__gxx_personality_v0(...)
declare i32 @__CxxFrameHandler3(...)

define void @test_simple_ret(ptr %p, i32 %v) {
; CHECK-LABEL: @test_simple_ret(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0:![0-9]+]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    ret void
;
  store i32 %v, ptr %p, align 4, !nontemporal !0
  ret void
}

define void @test_normal_store(ptr %p, i32 %v) {
; CHECK-LABEL: @test_normal_store(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4
; CHECK-NEXT:    ret void
;
  store i32 %v, ptr %p, align 4
  ret void
}

define void @test_multiple_stores(ptr %p, i32 %v1, i32 %v2) {
; CHECK-LABEL: @test_multiple_stores(
; CHECK-NEXT:    store i32 [[V1:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    store i32 [[V2:%.*]], ptr [[P]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    ret void
;
  store i32 %v1, ptr %p, align 4, !nontemporal !0
  store i32 %v2, ptr %p, align 4, !nontemporal !0
  ret void
}

define void @test_call_sync(ptr %p, i32 %v) {
; CHECK-LABEL: @test_call_sync(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    call void @sync_call()
; CHECK-NEXT:    call void @sync_call()
; CHECK-NEXT:    ret void
;
  store i32 %v, ptr %p, align 4, !nontemporal !0
  call void @sync_call()
  call void @sync_call()
  ret void
}

define void @test_call_nosync_nounwind(ptr %p, i32 %v) {
; CHECK-LABEL: @test_call_nosync_nounwind(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @nosync_nounwind_call()
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    ret void
;
  store i32 %v, ptr %p, align 4, !nontemporal !0
  call void @nosync_nounwind_call()
  ret void
}

define void @test_call_nosync_may_throw(ptr %p, i32 %v) {
; CHECK-LABEL: @test_call_nosync_may_throw(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    call void @nosync_call()
; CHECK-NEXT:    ret void
;
  store i32 %v, ptr %p, align 4, !nontemporal !0
  call void @nosync_call()
  ret void
}

define void @test_existing_sfence(ptr %p, i32 %v) {
; CHECK-LABEL: @test_existing_sfence(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    call void @sync_call()
; CHECK-NEXT:    ret void
;
  store i32 %v, ptr %p, align 4, !nontemporal !0
  call void @llvm.x86.sse.sfence()
  call void @sync_call()
  ret void
}

define void @test_existing_mfence(ptr %p, i32 %v) {
; CHECK-LABEL: @test_existing_mfence(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse2.mfence()
; CHECK-NEXT:    call void @sync_call()
; CHECK-NEXT:    ret void
;
  store i32 %v, ptr %p, align 4, !nontemporal !0
  call void @llvm.x86.sse2.mfence()
  call void @sync_call()
  ret void
}

define void @test_branch_merge(ptr %p, i32 %v, i1 %c) {
; CHECK-LABEL: @test_branch_merge(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    br label [[MERGE:%.*]]
; CHECK:       else:
; CHECK-NEXT:    br label [[MERGE]]
; CHECK:       merge:
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    ret void
;
entry:
  br i1 %c, label %then, label %else

then:
  store i32 %v, ptr %p, align 4, !nontemporal !0
  br label %merge

else:
  br label %merge

merge:
  ret void
}

define void @test_branch_both_return(ptr %p, i32 %v, i1 %c) {
; CHECK-LABEL: @test_branch_both_return(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    br i1 [[C:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    ret void
; CHECK:       bb2:
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    ret void
;
entry:
  store i32 %v, ptr %p, align 4, !nontemporal !0
  br i1 %c, label %bb1, label %bb2

bb1:
  ret void

bb2:
  ret void
}

define void @test_loop(ptr %p, i32 %v, i1 %c) {
; CHECK-LABEL: @test_loop(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    call void @sync_call()
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    br i1 [[C:%.*]], label [[LOOP]], label [[EXIT:%.*]]
; CHECK:       exit:
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    ret void
;
entry:
  br label %loop

loop:
  call void @sync_call()
  store i32 %v, ptr %p, align 4, !nontemporal !0
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

define void @test_loop_no_fence_inside(ptr %p, i32 %v, i32 %n) {
; CHECK-LABEL: @test_loop_no_fence_inside(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %entry ], [ [[I_NEXT:%.*]], %loop ]
; CHECK-NEXT:    [[P_GEP:%.*]] = getelementptr inbounds i32, ptr [[P:%.*]], i32 [[I]]
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P_GEP]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    [[I_NEXT]] = add i32 [[I]], 1
; CHECK-NEXT:    [[COND:%.*]] = icmp slt i32 [[I_NEXT]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[COND]], label [[LOOP]], label [[EXIT:%.*]]
; CHECK:       exit:
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    ret void
;
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %p.gep = getelementptr inbounds i32, ptr %p, i32 %i
  store i32 %v, ptr %p.gep, align 4, !nontemporal !0
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

define void @test_atomic_release(ptr %p, i32 %v, ptr %flag) {
; CHECK-LABEL: @test_atomic_release(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    store atomic i32 1, ptr [[FLAG:%.*]] release, align 4
; CHECK-NEXT:    ret void
;
  store i32 %v, ptr %p, align 4, !nontemporal !0
  store atomic i32 1, ptr %flag release, align 4
  ret void
}

define void @test_atomic_fence(ptr %p, i32 %v) {
; CHECK-LABEL: @test_atomic_fence(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    fence release
; CHECK-NEXT:    ret void
;
  store i32 %v, ptr %p, align 4, !nontemporal !0
  fence release
  ret void
}

define void @test_invoke(ptr %p, i32 %v) personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @test_invoke(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    invoke void @sync_call()
; CHECK-NEXT:            to label [[CONT:%.*]] unwind label [[LPAD:%.*]]
; CHECK:       cont:
; CHECK-NEXT:    ret void
; CHECK:       lpad:
; CHECK-NEXT:    [[LP:%.*]] = landingpad { ptr, i32 }
; CHECK-NEXT:            cleanup
; CHECK-NEXT:    resume { ptr, i32 } [[LP]]
;
entry:
  store i32 %v, ptr %p, align 4, !nontemporal !0
  invoke void @sync_call()
          to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %lp
}

define void @test_invoke_nosync(ptr %p, i32 %v) personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @test_invoke_nosync(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    invoke void @nosync_call()
; CHECK-NEXT:            to label [[CONT:%.*]] unwind label [[LPAD:%.*]]
; CHECK:       cont:
; CHECK-NEXT:    ret void
; CHECK:       lpad:
; CHECK-NEXT:    [[LP:%.*]] = landingpad { ptr, i32 }
; CHECK-NEXT:            cleanup
; CHECK-NEXT:    resume { ptr, i32 } [[LP]]
;
entry:
  store i32 %v, ptr %p, align 4, !nontemporal !0
  invoke void @nosync_call()
          to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %lp
}

define void @test_atomic_seq_cst_fence(ptr %p, i32 %v) {
; CHECK-LABEL: @test_atomic_seq_cst_fence(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    fence seq_cst
; CHECK-NEXT:    ret void
;
  store i32 %v, ptr %p, align 4, !nontemporal !0
  fence seq_cst
  ret void
}

define void @test_atomic_rmw(ptr %p, i32 %v, ptr %flag) {
; CHECK-LABEL: @test_atomic_rmw(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    [[PREV:%.*]] = atomicrmw add ptr [[FLAG:%.*]], i32 1 seq_cst, align 4
; CHECK-NEXT:    ret void
;
  store i32 %v, ptr %p, align 4, !nontemporal !0
  %prev = atomicrmw add ptr %flag, i32 1 seq_cst, align 4
  ret void
}

define void @test_atomic_cmpxchg(ptr %p, i32 %v, ptr %flag) {
; CHECK-LABEL: @test_atomic_cmpxchg(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    [[RES:%.*]] = cmpxchg ptr [[FLAG:%.*]], i32 0, i32 1 seq_cst seq_cst, align 4
; CHECK-NEXT:    ret void
;
  store i32 %v, ptr %p, align 4, !nontemporal !0
  %res = cmpxchg ptr %flag, i32 0, i32 1 seq_cst seq_cst, align 4
  ret void
}

define i32 @test_atomic_load_acquire(ptr %p, i32 %v, ptr %flag) {
; CHECK-LABEL: @test_atomic_load_acquire(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    [[VAL:%.*]] = load atomic i32, ptr [[FLAG:%.*]] acquire, align 4
; CHECK-NEXT:    ret i32 [[VAL]]
;
  store i32 %v, ptr %p, align 4, !nontemporal !0
  %val = load atomic i32, ptr %flag acquire, align 4
  ret i32 %val
}

define void @test_no_sse(ptr %p, i32 %v) {
; CHECK-LABEL: @test_no_sse(
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    ret void
;
; CHECK-NOSSE-LABEL: @test_no_sse(
; CHECK-NOSSE-NEXT:    store i32 %v, ptr %p, align 4, !nontemporal [[META0:![0-9]+]]
; CHECK-NOSSE-NEXT:    ret void
;
  store i32 %v, ptr %p, align 4, !nontemporal !0
  ret void
}

define void @test_catchswitch(ptr %p, i32 %v) personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: @test_catchswitch(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    store i32 [[V:%.*]], ptr [[P:%.*]], align 4, !nontemporal [[META0]]
; CHECK-NEXT:    call void @llvm.x86.sse.sfence()
; CHECK-NEXT:    invoke void @nosync_call()
; CHECK-NEXT:            to label [[CONT:%.*]] unwind label [[CS_BB:%.*]]
; CHECK:       cont:
; CHECK-NEXT:    ret void
; CHECK:       cs_bb:
; CHECK-NEXT:    [[CS:%.*]] = catchswitch within none [label %catch_bb] unwind to caller
; CHECK:       catch_bb:
; CHECK-NEXT:    [[CP:%.*]] = catchpad within [[CS]] []
; CHECK-NEXT:    catchret from [[CP]] to label [[CONT]]
;
entry:
  store i32 %v, ptr %p, align 4, !nontemporal !0
  invoke void @nosync_call()
          to label %cont unwind label %cs_bb

cont:
  ret void

cs_bb:
  %cs = catchswitch within none [label %catch_bb] unwind to caller

catch_bb:
  %cp = catchpad within %cs []
  catchret from %cp to label %cont
}

!0 = !{i32 1}
