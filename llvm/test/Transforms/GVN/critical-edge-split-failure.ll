; RUN: opt -passes=gvn -S -o - %s | FileCheck %s

%struct.sk_buff = type opaque

@l2tp_recv_dequeue_session = external dso_local local_unnamed_addr global i32, align 4
@l2tp_recv_dequeue_skb = external dso_local local_unnamed_addr global ptr, align 8
@l2tp_recv_dequeue_session_2 = external dso_local local_unnamed_addr global i32, align 4
@l2tp_recv_dequeue_session_0 = external dso_local local_unnamed_addr global i32, align 4

declare void @llvm.assume(i1 noundef)

define dso_local void @l2tp_recv_dequeue() local_unnamed_addr {
entry:
  %0 = load i32, ptr @l2tp_recv_dequeue_session, align 4
  %conv = sext i32 %0 to i64
  %1 = inttoptr i64 %conv to ptr
  %2 = load i32, ptr @l2tp_recv_dequeue_session_2, align 4
  %tobool.not = icmp eq i32 %2, 0
  br label %for.cond

for.cond:                                         ; preds = %if.end, %entry
  %storemerge = phi ptr [ %1, %entry ], [ null, %if.end ]
  store ptr %storemerge, ptr @l2tp_recv_dequeue_skb, align 8
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %for.cond
  %3 = load i32, ptr %storemerge, align 4
  store i32 %3, ptr @l2tp_recv_dequeue_session_0, align 4
; Splitting the critical edge from if.then to if.end will fail, but should not
; cause an infinite loop in GVN. If we can one day split edges of callbr
; indirect targets, great!
; CHECK: callbr void asm sideeffect "", "!i,~{dirflag},~{fpsr},~{flags}"()
; CHECK-NEXT: to label %asm.fallthrough.i [label %if.end]
  callbr void asm sideeffect "", "!i,~{dirflag},~{fpsr},~{flags}"()
          to label %asm.fallthrough.i [label %if.end]

asm.fallthrough.i:                                ; preds = %if.then
  br label %if.end

if.end:                                           ; preds = %asm.fallthrough.i, %if.then, %for.cond
  %4 = load i32, ptr %storemerge, align 4
  %tobool2.not = icmp eq i32 %4, 0
  tail call void @llvm.assume(i1 %tobool2.not)
  br label %for.cond
}

