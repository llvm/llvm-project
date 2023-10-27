; RUN: opt -passes='hardware-loops<force-hardware-loops>' -S %s -o - | FileCheck %s --check-prefix=CHECK
; RUN: opt -passes='hardware-loops<force-hardware-loops;hardware-loop-decrement=1>' -S %s -o - | FileCheck %s --check-prefix=CHECK
; RUN: opt -passes='hardware-loops<force-hardware-loops;hardware-loop-counter-bitwidth=32>' -S %s -o - | FileCheck %s --check-prefix=CHECK

define void @while_lt(i32 %i, i32 %N, ptr nocapture %A) {
; CHECK-LABEL: @while_lt(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP4:%.*]] = icmp ult i32 [[I:%.*]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[CMP4]], label [[WHILE_BODY_PREHEADER:%.*]], label [[WHILE_END:%.*]]
; CHECK:       while.body.preheader:
; CHECK-NEXT:    [[TMP0:%.*]] = sub i32 [[N]], [[I]]
; CHECK-NEXT:    call void @llvm.set.loop.iterations.i32(i32 [[TMP0]])
; CHECK-NEXT:    br label [[WHILE_BODY:%.*]]
; CHECK:       while.body:
; CHECK-NEXT:    [[I_ADDR_05:%.*]] = phi i32 [ [[INC:%.*]], [[WHILE_BODY]] ], [ [[I]], [[WHILE_BODY_PREHEADER]] ]
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr [[A:%.*]], i32 [[I_ADDR_05]]
; CHECK-NEXT:    store i32 [[I_ADDR_05]], ptr [[ARRAYIDX]], align 4
; CHECK-NEXT:    [[INC]] = add nuw i32 [[I_ADDR_05]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = call i1 @llvm.loop.decrement.i32(i32 1)
; CHECK-NEXT:    br i1 [[TMP1]], label [[WHILE_BODY]], label [[WHILE_END]]
; CHECK:       while.end:
; CHECK-NEXT:    ret void
entry:
  %cmp4 = icmp ult i32 %i, %N
  br i1 %cmp4, label %while.body, label %while.end

while.body:
  %i.addr.05 = phi i32 [ %inc, %while.body ], [ %i, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i.addr.05
  store i32 %i.addr.05, ptr %arrayidx, align 4
  %inc = add nuw i32 %i.addr.05, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %while.end, label %while.body

while.end:
  ret void
}
