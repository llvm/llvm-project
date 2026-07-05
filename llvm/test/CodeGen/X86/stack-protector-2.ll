; RUN: llc -mtriple=x86_64-pc-linux-gnu -start-before=stack-protector \
; RUN:   -stop-after=stack-protector -o - < %s | FileCheck %s
; Bugs 42238/43308: Test some additional situations not caught previously.

define void @store_captures() #0 {
; CHECK-LABEL: @store_captures(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[STACKGUARDSLOT:%.*]] = alloca ptr
; CHECK-NEXT:    [[STACKGUARD:%.*]] = load volatile ptr, ptr addrspace(257) inttoptr (i32 40 to ptr addrspace(257))
; CHECK-NEXT:    call void @llvm.stackprotector(ptr [[STACKGUARD]], ptr [[STACKGUARDSLOT]])
; CHECK-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[J:%.*]] = alloca ptr, align 8
; CHECK-NEXT:    store i32 0, ptr [[RETVAL]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, ptr [[A]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[LOAD]], 1
; CHECK-NEXT:    store i32 [[ADD]], ptr [[A]], align 4
; CHECK-NEXT:    store ptr [[A]], ptr [[J]], align 8
; CHECK-NEXT:    [[STACKGUARD1:%.*]] = load volatile ptr, ptr addrspace(257) inttoptr (i32 40 to ptr addrspace(257))
; CHECK-NEXT:    [[TMP0:%.*]] = load volatile ptr, ptr [[STACKGUARDSLOT]]
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[STACKGUARD1]], [[TMP0]]
; CHECK-NEXT:    br i1 [[TMP1]], label [[SP_RETURN:%.*]], label [[CALLSTACKCHECKFAILBLK:%.*]], !prof !0
; CHECK:       SP_return:
; CHECK-NEXT:    ret void
; CHECK:       CallStackCheckFailBlk:
; CHECK-NEXT:    call void @__stack_chk_fail()
; CHECK-NEXT:    unreachable
;
entry:
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %j = alloca ptr, align 8
  store i32 0, ptr %retval
  %load = load i32, ptr %a, align 4
  %add = add nsw i32 %load, 1
  store i32 %add, ptr %a, align 4
  store ptr %a, ptr %j, align 8
  ret void
}

define ptr @non_captures() #0 {
; load, atomicrmw, and ret do not trigger a stack protector.
; CHECK-LABEL: @non_captures(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, ptr [[A]], align 4
; CHECK-NEXT:    [[ATOM:%.*]] = atomicrmw add ptr [[A]], i32 1 seq_cst
; CHECK-NEXT:    ret ptr [[A]]
;
entry:
  %a = alloca i32, align 4
  %load = load i32, ptr %a, align 4
  %atom = atomicrmw add ptr %a, i32 1 seq_cst
  ret ptr %a
}

define void @store_addrspacecast_captures() #0 {
; CHECK-LABEL: @store_addrspacecast_captures(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[STACKGUARDSLOT:%.*]] = alloca ptr
; CHECK-NEXT:    [[STACKGUARD:%.*]] = load volatile ptr, ptr addrspace(257) inttoptr (i32 40 to ptr addrspace(257))
; CHECK-NEXT:    call void @llvm.stackprotector(ptr [[STACKGUARD]], ptr [[STACKGUARDSLOT]])
; CHECK-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[J:%.*]] = alloca ptr addrspace(1), align 8
; CHECK-NEXT:    store i32 0, ptr [[RETVAL]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, ptr [[A]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[LOAD]], 1
; CHECK-NEXT:    store i32 [[ADD]], ptr [[A]], align 4
; CHECK-NEXT:    [[A_ADDRSPACECAST:%.*]] = addrspacecast ptr [[A]] to ptr addrspace(1)
; CHECK-NEXT:    store ptr addrspace(1) [[A_ADDRSPACECAST]], ptr [[J]], align 8
; CHECK-NEXT:    [[STACKGUARD1:%.*]] = load volatile ptr, ptr addrspace(257) inttoptr (i32 40 to ptr addrspace(257))
; CHECK-NEXT:    [[TMP0:%.*]] = load volatile ptr, ptr [[STACKGUARDSLOT]]
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[STACKGUARD1]], [[TMP0]]
; CHECK-NEXT:    br i1 [[TMP1]], label [[SP_RETURN:%.*]], label [[CALLSTACKCHECKFAILBLK:%.*]], !prof !0
; CHECK:       SP_return:
; CHECK-NEXT:    ret void
; CHECK:       CallStackCheckFailBlk:
; CHECK-NEXT:    call void @__stack_chk_fail()
; CHECK-NEXT:    unreachable
;
entry:
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %j = alloca ptr addrspace(1), align 8
  store i32 0, ptr %retval
  %load = load i32, ptr %a, align 4
  %add = add nsw i32 %load, 1
  store i32 %add, ptr %a, align 4
  %a.addrspacecast = addrspacecast ptr %a to ptr addrspace(1)
  store ptr addrspace(1) %a.addrspacecast, ptr %j, align 8
  ret void
}

define void @cmpxchg_captures() #0 {
; CHECK-LABEL: @cmpxchg_captures(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[STACKGUARDSLOT:%.*]] = alloca ptr
; CHECK-NEXT:    [[STACKGUARD:%.*]] = load volatile ptr, ptr addrspace(257) inttoptr (i32 40 to ptr addrspace(257))
; CHECK-NEXT:    call void @llvm.stackprotector(ptr [[STACKGUARD]], ptr [[STACKGUARDSLOT]])
; CHECK-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[J:%.*]] = alloca ptr, align 8
; CHECK-NEXT:    store i32 0, ptr [[RETVAL]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, ptr [[A]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[LOAD]], 1
; CHECK-NEXT:    store i32 [[ADD]], ptr [[A]], align 4
; CHECK-NEXT:    [[TMP0:%.*]] = cmpxchg ptr [[J]], ptr null, ptr [[A]] seq_cst monotonic
; CHECK-NEXT:    [[STACKGUARD1:%.*]] = load volatile ptr, ptr addrspace(257) inttoptr (i32 40 to ptr addrspace(257))
; CHECK-NEXT:    [[TMP1:%.*]] = load volatile ptr, ptr [[STACKGUARDSLOT]]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp eq ptr [[STACKGUARD1]], [[TMP1]]
; CHECK-NEXT:    br i1 [[TMP2]], label [[SP_RETURN:%.*]], label [[CALLSTACKCHECKFAILBLK:%.*]], !prof !0
; CHECK:       SP_return:
; CHECK-NEXT:    ret void
; CHECK:       CallStackCheckFailBlk:
; CHECK-NEXT:    call void @__stack_chk_fail()
; CHECK-NEXT:    unreachable
;
entry:
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %j = alloca ptr, align 8
  store i32 0, ptr %retval
  %load = load i32, ptr %a, align 4
  %add = add nsw i32 %load, 1
  store i32 %add, ptr %a, align 4

  cmpxchg ptr %j, ptr null, ptr %a seq_cst monotonic
  ret void
}

define void @memset_captures(i64 %c) #0 {
; CHECK-LABEL: @memset_captures(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[STACKGUARDSLOT:%.*]] = alloca ptr
; CHECK-NEXT:    [[STACKGUARD:%.*]] = load volatile ptr, ptr addrspace(257) inttoptr (i32 40 to ptr addrspace(257))
; CHECK-NEXT:    call void @llvm.stackprotector(ptr [[STACKGUARD]], ptr [[STACKGUARDSLOT]])
; CHECK-NEXT:    [[CADDR:%.*]] = alloca i64, align 8
; CHECK-NEXT:    store i64 %c, ptr [[CADDR]], align 8
; CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[COUNT:%.*]] = load i64, ptr [[CADDR]], align 8
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 4 [[I]], i8 0, i64 [[COUNT]], i1 false)
; CHECK-NEXT:    [[STACKGUARD1:%.*]] = load volatile ptr, ptr addrspace(257) inttoptr (i32 40 to ptr addrspace(257))
; CHECK-NEXT:    [[TMP1:%.*]] = load volatile ptr, ptr [[STACKGUARDSLOT]]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp eq ptr [[STACKGUARD1]], [[TMP1]]
; CHECK-NEXT:    br i1 [[TMP2]], label [[SP_RETURN:%.*]], label [[CALLSTACKCHECKFAILBLK:%.*]], !prof !0
; CHECK:       SP_return:
; CHECK-NEXT:    ret void
; CHECK:       CallStackCheckFailBlk:
; CHECK-NEXT:    call void @__stack_chk_fail()
; CHECK-NEXT:    unreachable
;
entry:
  %c.addr = alloca i64, align 8
  store i64 %c, ptr %c.addr, align 8
  %i = alloca i32, align 4
  %count = load i64, ptr %c.addr, align 8
  call void @llvm.memset.p0.i64(ptr align 4 %i, i8 0, i64 %count, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

; Intentionally does not have any fn attrs.
declare dso_local void @foo(ptr)

; @bar_sspstrong and @bar_nossp are the same function, but differ only in
; function attributes. Test that a callee without stack protector function
; attribute does not trigger a stack guard slot in a caller that also does not
; have a stack protector slot.
define dso_local void @bar_sspstrong(i64 %0) #0 {
; CHECK-LABEL: @bar_sspstrong
; CHECK-NEXT: %StackGuardSlot = alloca ptr
  %2 = alloca i64, align 8
  store i64 %0, ptr %2, align 8
  %3 = load i64, ptr %2, align 8
  %4 = alloca i8, i64 %3, align 16
  call void @foo(ptr %4)
  ret void
}

; Intentionally does not have any fn attrs.
define dso_local void @bar_nossp(i64 %0) {
; CHECK-LABEL: @bar_nossp
; CHECK-NEXT: %2 = alloca i64
  %2 = alloca i64, align 8
  store i64 %0, ptr %2, align 8
  %3 = load i64, ptr %2, align 8
  %4 = alloca i8, i64 %3, align 16
  call void @foo(ptr %4)
  ret void
}

; Check stack protect for noreturn call
define dso_local i32 @foo_no_return(i32 %0) #1 {
; CHECK-LABEL: @foo_no_return
entry:
  %cmp = icmp sgt i32 %0, 4
  br i1 %cmp, label %if.then, label %if.end

; CHECK:      if.then:                                          ; preds = %entry
; CHECK-NEXT:   %StackGuard1 = load volatile ptr, ptr addrspace(257) inttoptr (i32 40 to ptr addrspace(257)), align 8
; CHECK-NEXT:   %1 = load volatile ptr, ptr %StackGuardSlot, align 8
; CHECK-NEXT:   %2 = icmp eq ptr %StackGuard1, %1
; CHECK-NEXT:   br i1 %2, label %SP_return, label %CallStackCheckFailBlk
; CHECK:      SP_return:                                        ; preds = %if.then
; CHECK-NEXT:   %call = call i32 @foo_no_return(i32 1)
; CHECK-NEXT:   br label %return
; CHECK:      if.end:                                           ; preds = %entry
; CHECK-NEXT:   br label %return

if.then:                                          ; preds = %entry
  %call = call i32 @foo_no_return(i32 1)
  br label %return

if.end:                                           ; preds = %entry
  br label %return

return:                                           ; preds = %if.end, %if.then
  ret i32 0
}

declare void @callee() noreturn nounwind
define void @caller() sspstrong {
; Test that a stack protector is NOT inserted when we call nounwind functions.
; CHECK-LABEL: @caller
; CHECK-NEXT: call void @callee
  call void @callee() noreturn nounwind
  ret void
}

attributes #0 = { sspstrong }
attributes #1 = { noreturn sspreq}
