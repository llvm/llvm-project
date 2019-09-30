; RUN: llc -mtriple=x86_64-pc-linux-gnu -start-before=stack-protector -stop-after=stack-protector -o - < %s | FileCheck %s
; Bugs 42238/43308: Test some additional situations not caught previously.

define void @store_captures() #0 {
; CHECK-LABEL: @store_captures(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[STACKGUARDSLOT:%.*]] = alloca i8*
; CHECK-NEXT:    [[STACKGUARD:%.*]] = load volatile i8*, i8* addrspace(257)* inttoptr (i32 40 to i8* addrspace(257)*)
; CHECK-NEXT:    call void @llvm.stackprotector(i8* [[STACKGUARD]], i8** [[STACKGUARDSLOT]])
; CHECK-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[J:%.*]] = alloca i32*, align 8
; CHECK-NEXT:    store i32 0, i32* [[RETVAL]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, i32* [[A]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[LOAD]], 1
; CHECK-NEXT:    store i32 [[ADD]], i32* [[A]], align 4
; CHECK-NEXT:    store i32* [[A]], i32** [[J]], align 8
; CHECK-NEXT:    [[STACKGUARD1:%.*]] = load volatile i8*, i8* addrspace(257)* inttoptr (i32 40 to i8* addrspace(257)*)
; CHECK-NEXT:    [[TMP0:%.*]] = load volatile i8*, i8** [[STACKGUARDSLOT]]
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i8* [[STACKGUARD1]], [[TMP0]]
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
  %j = alloca i32*, align 8
  store i32 0, i32* %retval
  %load = load i32, i32* %a, align 4
  %add = add nsw i32 %load, 1
  store i32 %add, i32* %a, align 4
  store i32* %a, i32** %j, align 8
  ret void
}

define i32* @non_captures() #0 {
; load, atomicrmw, and ret do not trigger a stack protector.
; CHECK-LABEL: @non_captures(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, i32* [[A]], align 4
; CHECK-NEXT:    [[ATOM:%.*]] = atomicrmw add i32* [[A]], i32 1 seq_cst
; CHECK-NEXT:    ret i32* [[A]]
;
entry:
  %a = alloca i32, align 4
  %load = load i32, i32* %a, align 4
  %atom = atomicrmw add i32* %a, i32 1 seq_cst
  ret i32* %a
}

define void @store_addrspacecast_captures() #0 {
; CHECK-LABEL: @store_addrspacecast_captures(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[STACKGUARDSLOT:%.*]] = alloca i8*
; CHECK-NEXT:    [[STACKGUARD:%.*]] = load volatile i8*, i8* addrspace(257)* inttoptr (i32 40 to i8* addrspace(257)*)
; CHECK-NEXT:    call void @llvm.stackprotector(i8* [[STACKGUARD]], i8** [[STACKGUARDSLOT]])
; CHECK-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[J:%.*]] = alloca i32 addrspace(1)*, align 8
; CHECK-NEXT:    store i32 0, i32* [[RETVAL]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, i32* [[A]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[LOAD]], 1
; CHECK-NEXT:    store i32 [[ADD]], i32* [[A]], align 4
; CHECK-NEXT:    [[A_ADDRSPACECAST:%.*]] = addrspacecast i32* [[A]] to i32 addrspace(1)*
; CHECK-NEXT:    store i32 addrspace(1)* [[A_ADDRSPACECAST]], i32 addrspace(1)** [[J]], align 8
; CHECK-NEXT:    [[STACKGUARD1:%.*]] = load volatile i8*, i8* addrspace(257)* inttoptr (i32 40 to i8* addrspace(257)*)
; CHECK-NEXT:    [[TMP0:%.*]] = load volatile i8*, i8** [[STACKGUARDSLOT]]
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i8* [[STACKGUARD1]], [[TMP0]]
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
  %j = alloca i32 addrspace(1)*, align 8
  store i32 0, i32* %retval
  %load = load i32, i32* %a, align 4
  %add = add nsw i32 %load, 1
  store i32 %add, i32* %a, align 4
  %a.addrspacecast = addrspacecast i32* %a to i32 addrspace(1)*
  store i32 addrspace(1)* %a.addrspacecast, i32 addrspace(1)** %j, align 8
  ret void
}

define void @cmpxchg_captures() #0 {
; CHECK-LABEL: @cmpxchg_captures(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[STACKGUARDSLOT:%.*]] = alloca i8*
; CHECK-NEXT:    [[STACKGUARD:%.*]] = load volatile i8*, i8* addrspace(257)* inttoptr (i32 40 to i8* addrspace(257)*)
; CHECK-NEXT:    call void @llvm.stackprotector(i8* [[STACKGUARD]], i8** [[STACKGUARDSLOT]])
; CHECK-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[J:%.*]] = alloca i32*, align 8
; CHECK-NEXT:    store i32 0, i32* [[RETVAL]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, i32* [[A]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[LOAD]], 1
; CHECK-NEXT:    store i32 [[ADD]], i32* [[A]], align 4
; CHECK-NEXT:    [[TMP0:%.*]] = cmpxchg i32** [[J]], i32* null, i32* [[A]] seq_cst monotonic
; CHECK-NEXT:    [[STACKGUARD1:%.*]] = load volatile i8*, i8* addrspace(257)* inttoptr (i32 40 to i8* addrspace(257)*)
; CHECK-NEXT:    [[TMP1:%.*]] = load volatile i8*, i8** [[STACKGUARDSLOT]]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp eq i8* [[STACKGUARD1]], [[TMP1]]
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
  %j = alloca i32*, align 8
  store i32 0, i32* %retval
  %load = load i32, i32* %a, align 4
  %add = add nsw i32 %load, 1
  store i32 %add, i32* %a, align 4

  cmpxchg i32** %j, i32* null, i32* %a seq_cst monotonic
  ret void
}

define void @memset_captures(i64 %c) #0 {
; CHECK-LABEL: @memset_captures(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[STACKGUARDSLOT:%.*]] = alloca i8*
; CHECK-NEXT:    [[STACKGUARD:%.*]] = load volatile i8*, i8* addrspace(257)* inttoptr (i32 40 to i8* addrspace(257)*)
; CHECK-NEXT:    call void @llvm.stackprotector(i8* [[STACKGUARD]], i8** [[STACKGUARDSLOT]])
; CHECK-NEXT:    [[CADDR:%.*]] = alloca i64, align 8
; CHECK-NEXT:    store i64 %c, i64* [[CADDR]], align 8
; CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[IPTR:%.*]] = bitcast i32* [[I]] to i8*
; CHECK-NEXT:    [[COUNT:%.*]] = load i64, i64* [[CADDR]], align 8
; CHECK-NEXT:    call void @llvm.memset.p0i8.i64(i8* align 4 [[IPTR]], i8 0, i64 [[COUNT]], i1 false)
; CHECK-NEXT:    [[STACKGUARD1:%.*]] = load volatile i8*, i8* addrspace(257)* inttoptr (i32 40 to i8* addrspace(257)*)
; CHECK-NEXT:    [[TMP1:%.*]] = load volatile i8*, i8** [[STACKGUARDSLOT]]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp eq i8* [[STACKGUARD1]], [[TMP1]]
; CHECK-NEXT:    br i1 [[TMP2]], label [[SP_RETURN:%.*]], label [[CALLSTACKCHECKFAILBLK:%.*]], !prof !0
; CHECK:       SP_return:
; CHECK-NEXT:    ret void
; CHECK:       CallStackCheckFailBlk:
; CHECK-NEXT:    call void @__stack_chk_fail()
; CHECK-NEXT:    unreachable
;
entry:
  %c.addr = alloca i64, align 8
  store i64 %c, i64* %c.addr, align 8
  %i = alloca i32, align 4
  %i.ptr = bitcast i32* %i to i8*
  %count = load i64, i64* %c.addr, align 8
  call void @llvm.memset.p0i8.i64(i8* align 4 %i.ptr, i8 0, i64 %count, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg)

attributes #0 = { sspstrong }
