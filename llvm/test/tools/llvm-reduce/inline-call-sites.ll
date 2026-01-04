; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=inline-call-sites -reduce-callsite-inline-threshold=-1 --test FileCheck --test-arg --check-prefixes=CHECK,INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=RESULT,CHECK %s < %t

; RESULT: @gv = global [2 x ptr] [ptr @only_gv_user, ptr @simple_callee]
@gv = global [2 x ptr] [ptr @only_gv_user, ptr @simple_callee]

; RESULT: @indirectbr.L = internal unnamed_addr constant [3 x ptr] [ptr blockaddress(@callee_with_indirectbr, %L1), ptr blockaddress(@callee_with_indirectbr, %L2), ptr null], align 8
@indirectbr.L = internal unnamed_addr constant [3 x ptr] [ptr blockaddress(@callee_with_indirectbr, %L1), ptr blockaddress(@callee_with_indirectbr, %L2), ptr null], align 8


; CHECK-LABEL: define void @simple_callee(
; RESULT-NEXT: store i32 123, ptr %arg, align 4
; RESULT-NEXT: ret void
define void @simple_callee(ptr %arg) {
  store i32 123, ptr %arg
  ret void
}

; CHECK-LABEL: define void @simple_caller(
; RESULT-NEXT: store i32 123, ptr %outer.arg, align 4
; RESULT-NEXT: ret void
define void @simple_caller(ptr %outer.arg) {
  call void @simple_callee(ptr %outer.arg)
  ret void
}

; CHECK-LABEL: define void @multi_simple_caller(
; RESULT-NEXT: store i32 123, ptr %outer.arg, align 4
; RESULT-NEXT: store i32 123, ptr %outer.arg, align 4
; RESULT-NEXT: store i32 123, ptr null, align 4
; RESULT-NEXT: ret void
define void @multi_simple_caller(ptr %outer.arg) {
  call void @simple_callee(ptr %outer.arg)
  call void @simple_callee(ptr %outer.arg)
  call void @simple_callee(ptr null)
  ret void
}

; CHECK-LABEL: define void @only_gv_user(
; RESULT-NEXT: store i32 666, ptr %arg, align 4
; RESULT-NEXT: ret void
define void @only_gv_user(ptr %arg) {
  store i32 666, ptr %arg
  ret void
}

; CHECK-LABEL: define void @recursive(
; RESULT-NEXT: call void @recursive(ptr %arg)
; RESULT-NEXT: ret void
define void @recursive(ptr %arg) {
  call void @recursive(ptr %arg)
  ret void
}

; CHECK-LABEL: define void @recursive_with_wrong_callsite_type(
; RESULT-NEXT: call void @recursive_with_wrong_callsite_type(ptr %arg, i32 2)
; RESULT-NEXT: ret void
define void @recursive_with_wrong_callsite_type(ptr %arg) {
  call void @recursive_with_wrong_callsite_type(ptr %arg, i32 2)
  ret void
}

; CHECK-LABEL: define void @non_callee_use(
; RESULT-NEXT: store i32 567, ptr %arg, align 4
; RESULT-NEXT: ret void
define void @non_callee_use(ptr %arg) {
  store i32 567, ptr %arg
  ret void
}

declare void @extern_ptr_use(ptr)

; CHECK-LABEL: define void @non_callee_user(
; RESULT-NEXT: call void @extern_ptr_use(ptr @non_callee_use)
; RESULT-NEXT: ret void
define void @non_callee_user() {
  call void @extern_ptr_use(ptr @non_callee_use)
  ret void
}

; CHECK-LABEL: define void @non_call_inst_use(
define void @non_call_inst_use(ptr %arg) {
  store i32 999, ptr %arg
  ret void
}

; CHECK-LABEL: define void @non_call_inst_user(
; RESULT-NEXT: store ptr @non_call_inst_use, ptr %arg, align 8
; RESULT-NEXT: ret void
define void @non_call_inst_user(ptr %arg) {
  store ptr @non_call_inst_use, ptr %arg
  ret void
}

; CHECK-LABEL: define i32 @used_wrong_call_type(
; RESULT-NEXT: store i32 123, ptr %arg, align 4
; RESULT-NEXT: ret i32 8
define i32 @used_wrong_call_type(ptr %arg) {
  store i32 123, ptr %arg
  ret i32 8
}

; Inlining doesn't support the UB cases
; CHECK-LABEL: define void @use_wrong_call_type(
; RESULT-NEXT: call void @used_wrong_call_type(ptr %outer.arg)
; RESULT-NEXT: ret void
define void @use_wrong_call_type(ptr %outer.arg) {
  call void @used_wrong_call_type(ptr %outer.arg)
  ret void
}

; INTERESTING-LABEL: define void @incompatible_gc_callee(

; RESULT-LABEL: define void @incompatible_gc_callee(ptr %arg) gc "gc0" {
; RESULT-NEXT: store i32 10000, ptr %arg, align 4
; RESULT-NEXT: ret void
define void @incompatible_gc_callee(ptr %arg) gc "gc0" {
  store i32 10000, ptr %arg
  ret void
}

; INTERESTING-LABEL: define void @incompatible_gc_caller(

; RESULT-LABEL: define void @incompatible_gc_caller(ptr %outer.arg) gc "gc1" {
; RESULT-NEXT: call void @incompatible_gc_callee(ptr %outer.arg)
; RESULT-NEXT: ret void
define void @incompatible_gc_caller(ptr %outer.arg) gc "gc1" {
  call void @incompatible_gc_callee(ptr %outer.arg)
  ret void
}

; INTERESTING-LABEL: define void @propagate_callee_gc(

; RESULT-LABEL: define void @propagate_callee_gc(ptr %arg) gc "propagate-gc" {
; RESULT-NEXT: store i32 10000, ptr %arg, align 4
; RESULT-NEXT: ret void
define void @propagate_callee_gc(ptr %arg) gc "propagate-gc" {
  store i32 10000, ptr %arg
  ret void
}

; INTERESTING-LABEL: define void @propagate_caller_gc(

; RESULT-LABEL: define void @propagate_caller_gc(ptr %arg) gc "propagate-gc" {
; RESULT-NEXT: store i32 10000, ptr %arg, align 4
; RESULT-NEXT: ret void
define void @propagate_caller_gc(ptr %arg)  {
  call void @propagate_callee_gc(ptr %arg)
  ret void
}

declare i32 @__gxx_personality_v0(...)

; INTERESTING-LABEL: define void @propagate_callee_personality(

; RESULT-LABEL: define void @propagate_callee_personality(ptr %arg) personality ptr @__gxx_personality_v0 {
; RESULT-NEXT: store i32 2000, ptr %arg, align 4
; RESULT-NEXT: ret void
define void @propagate_callee_personality(ptr %arg) personality ptr @__gxx_personality_v0 {
  store i32 2000, ptr %arg
  ret void
}

; INTERESTING-LABEL: define void @propagate_caller_personality(

; RESULT-LABEL: define void @propagate_caller_personality(ptr %arg) personality ptr @__gxx_personality_v0 {
; RESULT-NEXT: store i32 2000, ptr %arg, align 4
; RESULT-NEXT: ret void
define void @propagate_caller_personality(ptr %arg)  {
  call void @propagate_callee_personality(ptr %arg)
  ret void
}

; CHECK-LABEL: define void @callee_with_indirectbr(
define void @callee_with_indirectbr() {
entry:
  br label %L1

L1:                                               ; preds = %entry, %L1
  %i = phi i32 [ 0, %entry ], [ %inc, %L1 ]
  %inc = add i32 %i, 1
  %idxprom = zext i32 %i to i64
  %arrayidx = getelementptr inbounds [3 x ptr], ptr @indirectbr.L, i64 0, i64 %idxprom
  %brtarget = load ptr, ptr %arrayidx, align 8
  indirectbr ptr %brtarget, [label %L1, label %L2]

L2:                                               ; preds = %L1
  ret void
}

; CHECK-LABEL: define void @calls_func_with_indirectbr(

; RESULT: L1.i:
; RESULT-NEXT: %i.i = phi i32 [ 0, %call ], [ %inc.i, %L1.i ]
; RESULT-NEXT: %inc.i = add i32 %i.i, 1
; RESULT-NEXT: %idxprom.i = zext i32 %i.i to i64
; RESULT-NEXT: %arrayidx.i = getelementptr inbounds [3 x ptr], ptr @indirectbr.L, i64 0, i64 %idxprom.i
; RESULT-NEXT: %brtarget.i = load ptr, ptr %arrayidx.i, align 8
; RESULT-NEXT: indirectbr ptr %brtarget.i, [label %L1.i, label %callee_with_indirectbr.exit]

define void @calls_func_with_indirectbr(i1 %arg0) {
entry:
  br i1 %arg0, label %call, label %ret

call:
  call void @callee_with_indirectbr()
  br label %ret

ret:
  ret void
}


; CHECK-LABEL: define ptr @callee_with_blockaddress_use(
; RESULT: L2:
; RESULT-NEXT: store ptr blockaddress(@callee_with_blockaddress_use, %L1), ptr %alloca, align 8
; RESULT-NEXT: store ptr blockaddress(@callee_with_blockaddress_use, %L2), ptr %alloca, align 8
; RESULT-NEXT: store ptr blockaddress(@callee_with_blockaddress_use, %L3), ptr %alloca, align 8
; RESULT-NEXT: %cond1 = load volatile i1, ptr addrspace(1) null
; RESULT-NEXT: br i1 %cond1, label %L1, label %L3
define ptr @callee_with_blockaddress_use() {
entry:
  %alloca = alloca ptr
  %cond0 = load volatile i1, ptr addrspace(1) null
  br i1 %cond0, label %L1, label %L2

L1:
  br label %L2

L2:
  ; reference an earlier block
  store ptr blockaddress(@callee_with_blockaddress_use, %L1), ptr %alloca

  ; reference the block itself from the block
  store ptr blockaddress(@callee_with_blockaddress_use, %L2), ptr %alloca

  ; reference a later block
  store ptr blockaddress(@callee_with_blockaddress_use, %L3), ptr %alloca

  %cond1 = load volatile i1, ptr addrspace(1) null
  br i1 %cond1, label %L1, label %L3

L3:
  %load = load ptr, ptr %alloca
  ret ptr %load
}

; FIXME: This is not correctly remapping the blockaddress use
; CHECK-LABEL: define void @calls_func_with_blockaddress_use(
; RESULT: entry:
; RESULT-NEXT: %alloca.i = alloca ptr, align 8
; RESULT-NEXT: store i32 1000, ptr null, align 4
; RESULT-NEXT: br i1 %arg0, label %call, label %ret

; RESULT: call:
; RESULT-NEXT: store i32 2000, ptr null, align 4
; RESULT-NEXT: call void @llvm.lifetime.start.p0(ptr %alloca.i)
; RESULT-NEXT: %cond0.i = load volatile i1, ptr addrspace(1) null, align 1
; RESULT-NEXT: br i1 %cond0.i, label %L1.i, label %L2.i

; RESULT: L1.i: ; preds = %L2.i, %call
; RESULT-NEXT: br label %L2.i

; RESULT: L2.i:                                             ; preds = %L1.i, %call
; RESULT-NEXT: store ptr blockaddress(@callee_with_blockaddress_use, %L1), ptr %alloca.i, align 8
; RESULT-NEXT:   store ptr blockaddress(@calls_func_with_blockaddress_use, %L2.i), ptr %alloca.i, align 8
; RESULT-NEXT: store ptr blockaddress(@callee_with_blockaddress_use, %L3), ptr %alloca.i, align 8
; RESULT-NEXT: %cond1.i = load volatile i1, ptr addrspace(1) null, align 1
; RESULT-NEXT: br i1 %cond1.i, label %L1.i, label %callee_with_blockaddress_use.exit

; RESULT: callee_with_blockaddress_use.exit:                ; preds = %L2.i
; RESULT-NEXT: %load.i = load ptr, ptr %alloca.i, align 8
; RESULT-NEXT: call void @llvm.lifetime.end.p0(ptr %alloca.i)
; RESULT-NEXT: store i32 3000, ptr null, align 4
; RESULT-NEXT: br label %ret

; RESULT: ret: ; preds = %callee_with_blockaddress_use.exit, %entry
; RESULT-NEXT: store i32 4000, ptr null, align 4
; RESULT-NEXT: ret void
define void @calls_func_with_blockaddress_use(i1 %arg0) {
entry:
  store i32 1000, ptr null
  br i1 %arg0, label %call, label %ret

call:
  store i32 2000, ptr null
  call ptr @callee_with_blockaddress_use()
  store i32 3000, ptr null
  br label %ret

ret:
  store i32 4000, ptr null
  ret void
}

; CHECK-LABEL: define void @callee_with_fallthrough_blockaddress_use(
; RESULT: L2:
; RESULT-NEXT: store ptr blockaddress(@callee_with_fallthrough_blockaddress_use, %L1), ptr %alloca, align 8
; RESULT-NEXT: store ptr blockaddress(@callee_with_fallthrough_blockaddress_use, %L2), ptr %alloca, align 8
; RESULT-NEXT: store ptr blockaddress(@callee_with_fallthrough_blockaddress_use, %L3), ptr %alloca, align 8
; RESULT-NEXT: br label %L3
define void @callee_with_fallthrough_blockaddress_use() {
entry:
  %alloca = alloca ptr
  br label %L1

L1:
  store i32 999, ptr null
  br label %L2

L2:                                               ; preds = %entry, %L1
  ; reference a block before this block
  store ptr blockaddress(@callee_with_fallthrough_blockaddress_use, %L1), ptr %alloca

  ; reference the block itself from the block
  store ptr blockaddress(@callee_with_fallthrough_blockaddress_use, %L2), ptr %alloca

  ; reference a block after this block
  store ptr blockaddress(@callee_with_fallthrough_blockaddress_use, %L3), ptr %alloca
  br label %L3

L3:                                               ; preds = %L1
  %load = load ptr, ptr %alloca
  ret void
}


; CHECK-LABEL: define void @calls_func_with_fallthrough_blockaddress_use(
; RESULT: entry:
; RESULT-NEXT: %alloca.i = alloca ptr, align 8
; RESULT-NEXT: store i32 1000, ptr null
; RESULT-NEXT: br i1 %arg0, label %call, label %ret

; RESULT: call:
; RESULT-NEXT: store i32 2000, ptr null, align 4
; RESULT-NEXT: call void @llvm.lifetime.start.p0(ptr %alloca.i)
; RESULT-NEXT: br label %L1.i

; RESULT: L1.i: ; preds = %call
; RESULT-NEXT: store i32 999, ptr null, align 4
; RESULT-NEXT: br label %L2.i

; RESULT: L2.i:
; RESULT-NEXT: store ptr blockaddress(@calls_func_with_fallthrough_blockaddress_use, %L1.i), ptr %alloca.i, align 8
; RESULT-NEXT: store ptr blockaddress(@calls_func_with_fallthrough_blockaddress_use, %L2.i), ptr %alloca.i, align 8
; RESULT-NEXT: store ptr blockaddress(@callee_with_fallthrough_blockaddress_use, %L3), ptr %alloca.i, align 8
; RESULT-NEXT: br label %callee_with_fallthrough_blockaddress_use.exit

; RESULT: callee_with_fallthrough_blockaddress_use.exit:    ; preds = %L2.i
; RESULT-NEXT: %load.i = load ptr, ptr %alloca.i, align 8
; RESULT-NEXT: call void @llvm.lifetime.end.p0(ptr %alloca.i)
; RESULT-NEXT: store i32 3000, ptr null, align 4
; RESULT-NEXT: br label %ret

; RESULT: ret:
; RESULT-NEXT: store i32 4000, ptr null, align 4
; RESULT-NEXT: ret void
define void @calls_func_with_fallthrough_blockaddress_use(i1 %arg0) {
entry:
  store i32 1000, ptr null
  br i1 %arg0, label %call, label %ret

call:
  store i32 2000, ptr null
  call void @callee_with_fallthrough_blockaddress_use()
  store i32 3000, ptr null
  br label %ret

ret:
  store i32 4000, ptr null
  ret void
}

declare i32 @extern_returns_twice() returns_twice

; CHECK-LABEL: define i32 @callee_returns_twice(
; RESULT-NEXT: %call = call i32 @extern_returns_twice()
; RESULT-NEXT: %add = add nsw i32 1, %call
; RESULT-NEXT: ret i32 %add
define i32 @callee_returns_twice() {
  %call = call i32 @extern_returns_twice()
  %add = add nsw i32 1, %call
  ret i32 %add
}

; CHECK-LABEL: define i32 @caller_returns_twice_calls_callee_returns_twice(
; RESULT-NEXT: %call.i = call i32 @extern_returns_twice()
; RESULT-NEXT: %add.i = add nsw i32 1, %call.i
; RESULT-NEXT: %add = add nsw i32 1, %add.i
; RESULT-NEXT: ret i32 %add
  define i32 @caller_returns_twice_calls_callee_returns_twice() returns_twice {
  %call = call i32 @callee_returns_twice()
  %add = add nsw i32 1, %call
  ret i32 %add
}

; Inliner usually blocks inlining of returns_twice functions into
; non-returns_twice functions
; CHECK-LABEL: define i32 @regular_caller_calls_callee_returns_twice() {
; RESULT-NEXT: %call.i = call i32 @extern_returns_twice()
; RESULT-NEXT: %add.i = add nsw i32 1, %call.i
; RESULT-NEXT: %add = add nsw i32 1, %add.i
; RESULT-NEXT: ret i32 %add
define i32 @regular_caller_calls_callee_returns_twice() {
  %call = call i32 @callee_returns_twice()
  %add = add nsw i32 1, %call
  ret i32 %add
}

; CHECK-LABEL: define void @caller_with_vastart(
; RESULT-NEXT: %ap = alloca ptr, align 4
; RESULT-NEXT: %ap2 = alloca ptr, align 4
; RESULT-NEXT: call void @llvm.va_start.p0(ptr nonnull %ap)
; RESULT-NEXT: call void @llvm.va_end.p0(ptr nonnull %ap)
; RESULT-NEXT: call void @llvm.va_start.p0(ptr nonnull %ap)
; RESULT-NEXT: call void @llvm.va_end.p0(ptr nonnull %ap)
; RESULT-NEXT: ret void
define void @caller_with_vastart(ptr noalias nocapture readnone %args, ...) {
  %ap = alloca ptr, align 4
  %ap2 = alloca ptr, align 4
  call void @llvm.va_start.p0(ptr nonnull %ap)
  call fastcc void @callee_with_vaend(ptr nonnull %ap)
  call void @llvm.va_start.p0(ptr nonnull %ap)
  call fastcc void @callee_with_vaend_alwaysinline(ptr nonnull %ap)
  ret void
}

; CHECK-LABEL: define fastcc void @callee_with_vaend(
; RESULT-NEXT: tail call void @llvm.va_end.p0(ptr %a)
; RESULT-NEXT: ret void
define fastcc void @callee_with_vaend(ptr %a) {
  tail call void @llvm.va_end.p0(ptr %a)
  ret void
}

; CHECK-LABEL: define internal fastcc void @callee_with_vaend_alwaysinline(
; RESULT-NEXT: tail call void @llvm.va_end.p0(ptr %a)
; RESULT-NEXT: ret void
define internal fastcc void @callee_with_vaend_alwaysinline(ptr %a) alwaysinline {
  tail call void @llvm.va_end.p0(ptr %a)
  ret void
}

; CHECK-LABEL: define i32 @callee_with_va_start(
define i32 @callee_with_va_start(ptr %a, ...) {
  %vargs = alloca ptr, align 8
  tail call void @llvm.va_start.p0(ptr %a)
  %va1 = va_arg ptr %vargs, i32
  call void @llvm.va_end(ptr %vargs)
  ret i32 %va1
}

; CHECK-LABEL: define i32 @callee_vastart_caller(
; RESULT-NEXT: %vargs.i = alloca ptr, align 8
; RESULT-NEXT: %ap = alloca ptr, align 4
; RESULT-NEXT: %b = load i32, ptr null, align 4
; RESULT-NEXT: call void @llvm.lifetime.start.p0(ptr %vargs.i)
; RESULT-NEXT: call void @llvm.va_start.p0(ptr nonnull %ap)
; RESULT-NEXT: %va1.i = va_arg ptr %vargs.i, i32
; RESULT-NEXT: call void @llvm.va_end.p0(ptr %vargs.i)
; RESULT-NEXT: call void @llvm.lifetime.end.p0(ptr %vargs.i)
; RESULT-NEXT: ret i32 %va1.i
define i32 @callee_vastart_caller(ptr noalias nocapture readnone %args, ...) {
  %ap = alloca ptr, align 4
  %b = load i32, ptr null
  %result = call i32 (ptr, ...) @callee_with_va_start(ptr nonnull %ap, i32 %b)
  ret i32 %result
}

declare void @llvm.localescape(...)

; CHECK-LABEL: define internal void @callee_uses_localrecover(
define internal void @callee_uses_localrecover(ptr %fp) {
  %a.i8 = call ptr @llvm.localrecover(ptr @callee_uses_localescape, ptr %fp, i32 0)
  store i32 42, ptr %a.i8
  ret void
}

; CHECK-LABEL: define i32 @callee_uses_localescape(
; RESULT-NEXT: %a = alloca i32, align 4
; RESULT-NEXT: call void (...) @llvm.localescape(ptr %a)
; RESULT-NEXT: %fp = call ptr @llvm.frameaddress.p0(i32 0)
; RESULT-NEXT: %a.i8.i = call ptr @llvm.localrecover(ptr @callee_uses_localescape, ptr %fp, i32 0)
; RESULT-NEXT: store i32 42, ptr %a.i8.i, align 4
; RESULT-NEXT: %r = load i32, ptr %a, align 4
; RESULT-NEXT: ret i32 %r
define i32 @callee_uses_localescape() alwaysinline {
  %a = alloca i32
  call void (...) @llvm.localescape(ptr %a)
  %fp = call ptr @llvm.frameaddress(i32 0)
  tail call void @callee_uses_localrecover(ptr %fp)
  %r = load i32, ptr %a
  ret i32 %r
}

; CHECK-LABEL: define i32 @callee_uses_localescape_caller(
; RESULT-NEXT: %a.i = alloca i32, align 4
; RESULT-NEXT: call void @llvm.lifetime.start.p0(ptr %a.i)
; RESULT-NEXT: call void (...) @llvm.localescape(ptr %a.i)
; RESULT-NEXT: %fp.i = call ptr @llvm.frameaddress.p0(i32 0)
; RESULT-NEXT: %a.i8.i.i = call ptr @llvm.localrecover(ptr @callee_uses_localescape, ptr %fp.i, i32 0)
; RESULT-NEXT: store i32 42, ptr %a.i8.i.i, align 4
; RESULT-NEXT: %r.i = load i32, ptr %a.i, align 4
; RESULT-NEXT: call void @llvm.lifetime.end.p0(ptr %a.i)
; RESULT-NEXT: ret i32 %r.i
define i32 @callee_uses_localescape_caller() {
  %r = tail call i32 @callee_uses_localescape()
  ret i32 %r
}

declare void @llvm.icall.branch.funnel(...)

; CHECK-LABEL: define void @callee_uses_branch_funnel(
; RESULT-NEXT: musttail call void (...) @llvm.icall.branch.funnel(...)
; RESULT-NEXT: ret void
define void @callee_uses_branch_funnel(...) {
  musttail call void (...) @llvm.icall.branch.funnel(...)
  ret void
}

; FIXME: This should fail the verifier after inlining
; CHECK-LABEL: define void @callee_branch_funnel_musttail_caller(
; RESULT-NEXT: call void (...) @llvm.icall.branch.funnel()
; RESULT-NEXT: ret void
define void @callee_branch_funnel_musttail_caller() {
  call void (...) @callee_uses_branch_funnel()
  ret void
}

; Ignore noinline on the callee function
; CHECK-LABEL: define void @noinline_callee(
; RESULT-NEXT: store i32 123, ptr %arg, align 4
; RESULT-NEXT: ret void
define void @noinline_callee(ptr %arg) {
  store i32 123, ptr %arg
  ret void
}

; CHECK-LABEL: define void @calls_noinline_func(
; RESULT-NEXT: store i32 123, ptr %outer.arg, align 4
; RESULT-NEXT: ret void
define void @calls_noinline_func(ptr %outer.arg) {
  call void @noinline_callee(ptr %outer.arg)
  ret void
}

; Ignore noinline on the callsite
; CHECK-LABEL: define void @calls_noinline_callsite(
; RESULT-NEXT: store i32 123, ptr %outer.arg, align 4
; RESULT-NEXT: ret void
define void @calls_noinline_callsite(ptr %outer.arg) {
  call void @simple_callee(ptr %outer.arg) noinline
  ret void
}

; Ignore optnone
; CHECK-LABEL: define void @optnone_callee(
; RESULT-NEXT: store i32 5555, ptr %arg, align 4
; RESULT-NEXT: ret void
define void @optnone_callee(ptr %arg) optnone noinline {
  store i32 5555, ptr %arg
  ret void
}

; CHECK-LABEL: define void @calls_optnone_callee(
; RESULT-NEXT: store i32 5555, ptr %outer.arg, align 4
; RESULT-NEXT: ret void
define void @calls_optnone_callee(ptr %outer.arg) {
  call void @optnone_callee(ptr %outer.arg)
  ret void
}

; CHECK-LABEL: define void @optnone_caller(
; RESULT-NEXT: store i32 123, ptr %outer.arg, align 4
; RESULT-NEXT: ret void
define void @optnone_caller(ptr %outer.arg) optnone noinline {
  call void @simple_callee(ptr %outer.arg)
  ret void
}

; CHECK-LABEL: define weak void @interposable_callee(
; RESULT-NEXT: store i32 2024, ptr %arg, align 4
; RESULT-NEXT: ret void
define weak void @interposable_callee(ptr %arg) {
  store i32 2024, ptr %arg
  ret void
}

; Ignore interposable linkage
; CHECK-LABEL: @calls_interposable_callee(
; RESULT-NEXT: store i32 2024, ptr %arg, align 4
; RESULT-NEXT: ret void
define void @calls_interposable_callee(ptr %arg) {
  call void @interposable_callee(ptr %arg)
  ret void
}

; Ignore null_pointer_is_valid
; CHECK-LABEL: @null_pointer_is_valid_callee(
; RESULT-NEXT: store i32 42069, ptr %arg, align 4
; RESULT-NEXT: ret void
define void @null_pointer_is_valid_callee(ptr %arg) null_pointer_is_valid {
  store i32 42069, ptr %arg
  ret void
}

; CHECK-LABEL: @calls_null_pointer_is_valid_callee(
; RESULT-NEXT: store i32 42069, ptr %arg, align 4
; RESULT-NEXT: ret void
define void @calls_null_pointer_is_valid_callee(ptr %arg) {
  call void @null_pointer_is_valid_callee(ptr %arg)
  ret void
}

; CHECK-LABEL: @byval_arg_uses_non_alloca_addrspace(
; RESULT-NEXT: %load = load i32, ptr addrspace(1) %arg, align 4
; RESULT-NEXT: ret i32 %load
define i32 @byval_arg_uses_non_alloca_addrspace(ptr addrspace(1) byval(i32) %arg) {
  %load = load i32, ptr addrspace(1) %arg
  ret i32 %load
}

; CHECK-LABEL: @calls_byval_arg_uses_non_alloca_addrspace(
; RESULT-NEXT: %arg1 = alloca i32, align 4, addrspace(1)
; RESULT-NEXT: call void @llvm.lifetime.start.p1(ptr addrspace(1) %arg1)
; RESULT-NEXT: call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 %arg1, ptr addrspace(1) %arg, i64 4, i1 false)
; RESULT-NEXT: %load.i = load i32, ptr addrspace(1) %arg1, align 4
; RESULT-NEXT: call void @llvm.lifetime.end.p1(ptr addrspace(1) %arg1)
; RESULT-NEXT: ret i32 %load.i
define i32 @calls_byval_arg_uses_non_alloca_addrspace(ptr addrspace(1) %arg) {
  %call = call i32 @byval_arg_uses_non_alloca_addrspace(ptr addrspace(1) byval(i32) %arg)
  ret i32 %call
}

; CHECK-LABEL: define void @callee_stacksize(
; RESULT-NEXT: %alloca = alloca [4096 x i32]
; RESULT-NEXT: store i32 12345678, ptr %arg
; RESULT-NEXT: store i32 0, ptr %alloca
; RESULT-NEXT: ret void
define void @callee_stacksize(ptr %arg) "inline-max-stacksize"="4" {
  %alloca = alloca [4096 x i32]
  store i32 12345678, ptr %arg
  store i32 0, ptr %alloca
  ret void
}

; CHECK-LABEL: define void @caller_stacksize(
; RESULT-NEXT: %alloca.i = alloca [4096 x i32], align 4
; RESULT-NEXT: call void @llvm.lifetime.start.p0(ptr %alloca.i)
; RESULT-NEXT: store i32 12345678, ptr %arg, align 4
; RESULT-NEXT: store i32 0, ptr %alloca.i, align 4
; RESULT-NEXT: call void @llvm.lifetime.end.p0(ptr %alloca.i)
; RESULT-NEXT: ret void
define void @caller_stacksize(ptr %arg) {
  call void @callee_stacksize(ptr %arg)
  ret void
}

; CHECK-LABEL: define void @callee_dynamic_alloca(
; RESULT-NEXT: %alloca = alloca i32, i32 %n, align 4
; RESULT-NEXT: store i32 12345678, ptr %arg, align 4
; RESULT-NEXT: store i32 0, ptr %alloca, align 4
; RESULT-NEXT: ret void
define void @callee_dynamic_alloca(ptr %arg, i32 %n) "inline-max-stacksize"="4" {
  %alloca = alloca i32, i32 %n
  store i32 12345678, ptr %arg
  store i32 0, ptr %alloca
  ret void
}

; CHECK-LABEL: define void @caller_dynamic_alloca(
; RESULT-NEXT: %savedstack = call ptr @llvm.stacksave.p0()
; RESULT-NEXT: %alloca.i = alloca i32, i32 %size, align 4
; RESULT-NEXT: store i32 12345678, ptr %arg, align 4
; RESULT-NEXT: store i32 0, ptr %alloca.i, align 4
; RESULT-NEXT: call void @llvm.stackrestore.p0(ptr %savedstack)
; RESULT-NEXT: ret void
define void @caller_dynamic_alloca(ptr %arg, i32 %size) {
  call void @callee_dynamic_alloca(ptr %arg, i32 %size)
  ret void
}

declare void @extern_noduplicate() noduplicate

; CHECK-LABEL: define void @callee_noduplicate_calls(
; RESULT-NEXT: call void @extern_noduplicate()
; RESULT-NEXT: call void @extern_noduplicate()
; RESULT-NEXT: ret void
define void @callee_noduplicate_calls() {
  call void @extern_noduplicate()
  call void @extern_noduplicate()
  ret void
}

; Ignore noduplicate restrictions
; CHECK-LABEL: define void @caller_noduplicate_calls_callee(
; RESULT-NEXT: call void @extern_noduplicate()
; RESULT-NEXT: call void @extern_noduplicate()
; RESULT-NEXT: call void @extern_noduplicate()
; RESULT-NEXT: call void @extern_noduplicate()
; RESULT-NEXT: ret void
define void @caller_noduplicate_calls_callee() {
  call void @callee_noduplicate_calls()
  call void @callee_noduplicate_calls()
  ret void
}

; CHECK-LABEL: define void @sanitize_address_callee(
; RESULT-NEXT: store i32 333, ptr %arg
; RESULT-NEXT: ret void
define void @sanitize_address_callee(ptr %arg) sanitize_address {
  store i32 333, ptr %arg
  ret void
}

; CHECK-LABEL: define void @no_sanitize_address_caller(
; RESULT-NEXT: store i32 333, ptr %arg
; RESULT-NEXT: ret void
define void @no_sanitize_address_caller(ptr %arg) {
  call void @sanitize_address_callee(ptr %arg)
  ret void
}

; CHECK-LABEL: define float @nonstrictfp_callee(
; RESULT-NEXT: %add = fadd float %a, %a
; RESULT-NEXT: ret float %add
define float @nonstrictfp_callee(float %a) {
  %add = fadd float %a, %a
  ret float %add
}

; CHECK-LABEL: define float @strictfp_caller(
; RESULT-NEXT: call float @llvm.experimental.constrained.fadd.f32(
; RESULT-NEXT: call float @llvm.experimental.constrained.fadd.f32(
; RESULT-NEXT: ret float %add
define float @strictfp_caller(float %a) strictfp {
  %call = call float @nonstrictfp_callee(float %a) strictfp
  %add = call float @llvm.experimental.constrained.fadd.f32(float %call, float 2.0, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %add
}

; CHECK-LABEL: define float @strictfp_callee(
; RESULT-NEXT: call float @llvm.experimental.constrained.fadd.f32(
; RESULT-NEXT: ret float
define float @strictfp_callee(float %a) strictfp {
  %add = call float @llvm.experimental.constrained.fadd.f32(float %a, float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %add
}

; FIXME: This should not inline. The inlined case should fail the
; verifier, but it does not.
; CHECK-LABEL: define float @nonstrictfp_caller(
; RESULT-NEXT: call float @llvm.experimental.constrained.fadd.f32(
; RESULT-NEXT: fadd float
; RESULT-NEXT: ret float
define float @nonstrictfp_caller(float %a) {
  %call = call float @strictfp_callee(float %a)
  %add1 = fadd float %call, 2.0
  ret float %add1
}

define void @caller_also_has_non_callee_use() {
  call void @simple_callee(ptr @simple_callee)
  ret void
}
