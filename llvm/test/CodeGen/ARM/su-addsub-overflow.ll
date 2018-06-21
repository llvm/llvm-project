; RUN: llc < %s -mtriple=arm-eabi -mcpu=generic | FileCheck %s

define i32 @sadd(i32 %a, i32 %b) local_unnamed_addr #0 {
; CHECK-LABEL: sadd:
; CHECK:    mov r[[R0:[0-9]+]], r0
; CHECK-NEXT:    add r[[R1:[0-9]+]], r[[R0]], r1
; CHECK-NEXT:    cmp r[[R1]], r[[R0]]
; CHECK-NEXT:    movvc pc, lr
entry:
  %0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:
  tail call void @llvm.trap() #2
  unreachable

cont:
  %2 = extractvalue { i32, i1 } %0, 0
  ret i32 %2

}

define i32 @uadd(i32 %a, i32 %b) local_unnamed_addr #0 {
; CHECK-LABEL: uadd:
; CHECK:    mov r[[R0:[0-9]+]], r0
; CHECK-NEXT:    adds r[[R1:[0-9]+]], r[[R0]], r1
; CHECK-NEXT:    cmp r[[R1]], r[[R0]]
; CHECK-NEXT:    movhs pc, lr
entry:
  %0 = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:
  tail call void @llvm.trap() #2
  unreachable

cont:
  %2 = extractvalue { i32, i1 } %0, 0
  ret i32 %2

}

define i32 @ssub(i32 %a, i32 %b) local_unnamed_addr #0 {
; CHECK-LABEL: ssub:
; CHECK:    cmp r0, r1
; CHECK-NEXT:    subvc r0, r0, r1
; CHECK-NEXT:    movvc pc, lr
entry:
  %0 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:
  tail call void @llvm.trap() #2
  unreachable

cont:
  %2 = extractvalue { i32, i1 } %0, 0
  ret i32 %2

}

define i32 @usub(i32 %a, i32 %b) local_unnamed_addr #0 {
; CHECK-LABEL: usub:
; CHECK:    mov r[[R0:[0-9]+]], r0
; CHECK-NEXT:    subs r[[R1:[0-9]+]], r[[R0]], r1
; CHECK-NEXT:    cmp r[[R0]], r1
; CHECK-NEXT:    movhs pc, lr
entry:
  %0 = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:
  tail call void @llvm.trap() #2
  unreachable

cont:
  %2 = extractvalue { i32, i1 } %0, 0
  ret i32 %2

}

define void @sum(i32* %a, i32* %b, i32 %n) local_unnamed_addr #0 {
; CHECK-LABEL: sum:
; CHECK:    ldr [[R0:r[0-9]+]],
; CHECK-NEXT:    ldr [[R1:r[0-9]+|lr]],
; CHECK-NEXT:    add [[R2:r[0-9]+]], [[R1]], [[R0]]
; CHECK-NEXT:    cmp [[R2]], [[R1]]
; CHECK-NEXT:    strvc [[R2]],
; CHECK-NEXT:    addvc
; CHECK-NEXT:    cmpvc
; CHECK-NEXT:    bvs
entry:
  %cmp7 = icmp eq i32 %n, 0
  br i1 %cmp7, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.08 = phi i32 [ %7, %cont2 ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i32 %i.08
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %a, i32 %i.08
  %1 = load i32, i32* %arrayidx1, align 4
  %2 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %1, i32 %0)
  %3 = extractvalue { i32, i1 } %2, 1
  br i1 %3, label %trap, label %cont

trap:
  tail call void @llvm.trap() #2
  unreachable

cont:
  %4 = extractvalue { i32, i1 } %2, 0
  store i32 %4, i32* %arrayidx1, align 4
  %5 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %i.08, i32 1)
  %6 = extractvalue { i32, i1 } %5, 1
  br i1 %6, label %trap, label %cont2

cont2:
  %7 = extractvalue { i32, i1 } %5, 0
  %cmp = icmp eq i32 %7, %n
  br i1 %cmp, label %for.cond.cleanup, label %for.body

}

declare void @llvm.trap() #2
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32) #1
