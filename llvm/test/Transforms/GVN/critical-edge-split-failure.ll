; RUN: opt -gvn -S -o - %s | FileCheck %s
; RUN: opt -passes=gvn -S -o - %s | FileCheck %s

%struct.sk_buff = type opaque

@l2tp_recv_dequeue_session = external dso_local local_unnamed_addr global i32, align 4
@l2tp_recv_dequeue_skb = external dso_local local_unnamed_addr global %struct.sk_buff*, align 8
@l2tp_recv_dequeue_session_2 = external dso_local local_unnamed_addr global i32, align 4
@l2tp_recv_dequeue_session_0 = external dso_local local_unnamed_addr global i32, align 4

declare void @llvm.assume(i1 noundef)

define dso_local void @l2tp_recv_dequeue() local_unnamed_addr {
entry:
  %0 = load i32, i32* @l2tp_recv_dequeue_session, align 4
  %conv = sext i32 %0 to i64
  %1 = inttoptr i64 %conv to %struct.sk_buff*
  %2 = load i32, i32* @l2tp_recv_dequeue_session_2, align 4
  %tobool.not = icmp eq i32 %2, 0
  br label %for.cond

for.cond:                                         ; preds = %if.end, %entry
  %storemerge = phi %struct.sk_buff* [ %1, %entry ], [ null, %if.end ]
  store %struct.sk_buff* %storemerge, %struct.sk_buff** @l2tp_recv_dequeue_skb, align 8
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %for.cond
  %ns = bitcast %struct.sk_buff* %storemerge to i32*
  %3 = load i32, i32* %ns, align 4
  store i32 %3, i32* @l2tp_recv_dequeue_session_0, align 4
; Splitting the critical edge from if.then to if.end will fail, but should not
; cause an infinite loop in GVN. If we can one day split edges of callbr
; indirect targets, great!
; CHECK: callbr void asm sideeffect "", "i,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@l2tp_recv_dequeue, %if.end))
; CHECK-NEXT: to label %asm.fallthrough.i [label %if.end]
  callbr void asm sideeffect "", "i,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@l2tp_recv_dequeue, %if.end))
          to label %asm.fallthrough.i [label %if.end]

asm.fallthrough.i:                                ; preds = %if.then
  br label %if.end

if.end:                                           ; preds = %asm.fallthrough.i, %if.then, %for.cond
  %ns1 = bitcast %struct.sk_buff* %storemerge to i32*
  %4 = load i32, i32* %ns1, align 4
  %tobool2.not = icmp eq i32 %4, 0
  tail call void @llvm.assume(i1 %tobool2.not)
  br label %for.cond
}

