; RUN: opt < %s -basic-aa -tbaa -licm -S | FileCheck %s
; RUN: opt -aa-pipeline=type-based-aa,basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' -S %s | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@X = global i32 7   ; <i32*> [#uses=4]

define void @test1(i32 %i) {
Entry:
  br label %Loop
; CHECK-LABEL: @test1(
; CHECK: Entry:
; CHECK-NEXT:   load i32, i32* @X
; CHECK-NEXT:   br label %Loop


Loop:   ; preds = %Loop, %0
  %j = phi i32 [ 0, %Entry ], [ %Next, %Loop ]    ; <i32> [#uses=1]
  %x = load i32, i32* @X   ; <i32> [#uses=1]
  %x2 = add i32 %x, 1   ; <i32> [#uses=1]
  store i32 %x2, i32* @X
  %Next = add i32 %j, 1   ; <i32> [#uses=2]
  %cond = icmp eq i32 %Next, 0    ; <i1> [#uses=1]
  br i1 %cond, label %Out, label %Loop

Out:
  ret void
; CHECK: Out:
; CHECK-NEXT:   %[[LCSSAPHI:.*]] = phi i32 [ %x2
; CHECK-NEXT:   store i32 %[[LCSSAPHI]], i32* @X
; CHECK-NEXT:   ret void

}

define void @test2(i32 %i) {
Entry:
  br label %Loop
; CHECK-LABEL: @test2(
; CHECK: Entry:
; CHECK-NEXT:    %.promoted = load i32, i32* getelementptr inbounds (i32, i32* @X, i64 1)
; CHECK-NEXT:    br label %Loop

Loop:   ; preds = %Loop, %0
  %X1 = getelementptr i32, i32* @X, i64 1    ; <i32*> [#uses=1]
  %A = load i32, i32* %X1    ; <i32> [#uses=1]
  %V = add i32 %A, 1    ; <i32> [#uses=1]
  %X2 = getelementptr i32, i32* @X, i64 1    ; <i32*> [#uses=1]
  store i32 %V, i32* %X2
  br i1 false, label %Loop, label %Exit

Exit:   ; preds = %Loop
  ret void
; CHECK: Exit:
; CHECK-NEXT:   %[[LCSSAPHI:.*]] = phi i32 [ %V
; CHECK-NEXT:   store i32 %[[LCSSAPHI]], i32* getelementptr inbounds (i32, i32* @X, i64 1)
; CHECK-NEXT:   ret void
}



define void @test3(i32 %i) {
; CHECK-LABEL: @test3(
  br label %Loop
Loop:
        ; Should not promote this to a register
  %x = load volatile i32, i32* @X
  %x2 = add i32 %x, 1
  store i32 %x2, i32* @X
  br i1 true, label %Out, label %Loop

; CHECK: Loop:
; CHECK-NEXT: load volatile

Out:    ; preds = %Loop
  ret void
}

define void @test3b(i32 %i) {
; CHECK-LABEL: @test3b(
; CHECK-LABEL: Loop:
; CHECK: store volatile
; CHECK-LABEL: Out:
  br label %Loop
Loop:
        ; Should not promote this to a register
  %x = load i32, i32* @X
  %x2 = add i32 %x, 1
  store volatile i32 %x2, i32* @X
  br i1 true, label %Out, label %Loop

Out:    ; preds = %Loop
  ret void
}

; PR8041
define void @test4(i8* %x, i8 %n) {
; CHECK-LABEL: @test4(
  %handle1 = alloca i8*
  %handle2 = alloca i8*
  store i8* %x, i8** %handle1
  br label %loop

loop:
  %tmp = getelementptr i8, i8* %x, i64 8
  store i8* %tmp, i8** %handle2
  br label %subloop

subloop:
  %count = phi i8 [ 0, %loop ], [ %nextcount, %subloop ]
  %offsetx2 = load i8*, i8** %handle2
  store i8 %n, i8* %offsetx2
  %newoffsetx2 = getelementptr i8, i8* %offsetx2, i64 -1
  store i8* %newoffsetx2, i8** %handle2
  %nextcount = add i8 %count, 1
  %innerexitcond = icmp sge i8 %nextcount, 8
  br i1 %innerexitcond, label %innerexit, label %subloop

; Should have promoted 'handle2' accesses.
; CHECK: subloop:
; CHECK-NEXT: phi i8* [
; CHECK-NEXT: %count = phi i8 [
; CHECK-NEXT: store i8 %n
; CHECK-NOT: store
; CHECK: br i1

innerexit:
  %offsetx1 = load i8*, i8** %handle1
  %val = load i8, i8* %offsetx1
  %cond = icmp eq i8 %val, %n
  br i1 %cond, label %exit, label %loop

; Should not have promoted offsetx1 loads.
; CHECK: innerexit:
; CHECK: %val = load i8, i8* %offsetx1
; CHECK: %cond = icmp eq i8 %val, %n
; CHECK: br i1 %cond, label %exit, label %loop

exit:
  ret void
}

define void @test5(i32 %i, i32** noalias %P2) {
Entry:
  br label %Loop
; CHECK-LABEL: @test5(
; CHECK: Entry:
; CHECK-NEXT:   load i32, i32* @X
; CHECK-NEXT:   br label %Loop


Loop:   ; preds = %Loop, %0
  %j = phi i32 [ 0, %Entry ], [ %Next, %Loop ]    ; <i32> [#uses=1]
  %x = load i32, i32* @X   ; <i32> [#uses=1]
  %x2 = add i32 %x, 1   ; <i32> [#uses=1]
  store i32 %x2, i32* @X

        store atomic i32* @X, i32** %P2 monotonic, align 8

  %Next = add i32 %j, 1   ; <i32> [#uses=2]
  %cond = icmp eq i32 %Next, 0    ; <i1> [#uses=1]
  br i1 %cond, label %Out, label %Loop

Out:
  ret void
; CHECK: Out:
; CHECK-NEXT:   %[[LCSSAPHI:.*]] = phi i32 [ %x2
; CHECK-NEXT:   store i32 %[[LCSSAPHI]], i32* @X
; CHECK-NEXT:   ret void

}


; PR14753 - Preserve TBAA tags when promoting values in a loop.
define void @test6(i32 %n, float* nocapture %a, i32* %gi) {
entry:
  store i32 0, i32* %gi, align 4, !tbaa !0
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %storemerge2 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %idxprom = sext i32 %storemerge2 to i64
  %arrayidx = getelementptr inbounds float, float* %a, i64 %idxprom
  store float 0.000000e+00, float* %arrayidx, align 4, !tbaa !3
  %0 = load i32, i32* %gi, align 4, !tbaa !0
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %gi, align 4, !tbaa !0
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void

; CHECK: for.body.lr.ph:
; CHECK-NEXT:  %gi.promoted = load i32, i32* %gi, align 4, !tbaa !0
; CHECK: for.cond.for.end_crit_edge:
; CHECK-NEXT:  %[[LCSSAPHI:.*]] = phi i32 [ %inc
; CHECK-NEXT:  store i32 %[[LCSSAPHI]], i32* %gi, align 4, !tbaa !0
}

declare i32 @opaque(i32) argmemonly
declare void @capture(i32*)

; We can promote even if opaque may throw.
define i32 @test7() {
; CHECK-LABEL: @test7(
; CHECK: entry:
; CHECK-NEXT: %local = alloca
; CHECK-NEXT: call void @capture(i32* %local)
; CHECK-NEXT: load i32, i32* %local
; CHECK-NEXT: br label %loop
; CHECK: exit:
; CHECK-NEXT: %[[LCSSAPHI:.*]] = phi i32 [ %x2, %loop ]
; CHECK-NEXT: store i32 %[[LCSSAPHI]], i32* %local
; CHECK-NEXT: %ret = load i32, i32* %local
; CHECK-NEXT: ret i32 %ret
entry:
  %local = alloca i32
  call void @capture(i32* %local)
  br label %loop

loop:
  %j = phi i32 [ 0, %entry ], [ %next, %loop ]
  %x = load i32, i32* %local
  %x2 = call i32 @opaque(i32 %x) ; Note this does not capture %local
  store i32 %x2, i32* %local
  %next = add i32 %j, 1
  %cond = icmp eq i32 %next, 0
  br i1 %cond, label %exit, label %loop

exit:
  %ret = load i32, i32* %local
  ret i32 %ret
}

; Make sure we don't promote if the store is really control-flow dependent.
define i32 @test7bad() {
; CHECK-LABEL: @test7bad(
; CHECK: entry:
; CHECK-NEXT: %local = alloca
; CHECK-NEXT: call void @capture(i32* %local)
; CHECK-NEXT: br label %loop
; CHECK: if:
; CHECK-NEXT: store i32 %x2, i32* %local
; CHECK-NEXT: br label %else
; CHECK: exit:
; CHECK-NEXT: %ret = load i32, i32* %local
; CHECK-NEXT: ret i32 %ret
entry:
  %local = alloca i32
  call void @capture(i32* %local)  
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %next, %else ]
  %x = load i32, i32* %local
  %x2 = call i32 @opaque(i32 %x) ; Note this does not capture %local
  %cmp = icmp eq i32 %x2, 0
  br i1 %cmp, label %if, label %else

if:  
  store i32 %x2, i32* %local
  br label %else

else:
  %next = add i32 %j, 1
  %cond = icmp eq i32 %next, 0
  br i1 %cond, label %exit, label %loop

exit:
  %ret = load i32, i32* %local
  ret i32 %ret
}

; Even if neither the load nor the store or guaranteed to execute because
; opaque() may throw, we can still promote - the load not being guaranteed
; doesn't block us, because %local is always dereferenceable.
define i32 @test8() {
; CHECK-LABEL: @test8(
; CHECK: entry:
; CHECK-NEXT: %local = alloca
; CHECK-NEXT: call void @capture(i32* %local)
; CHECK-NEXT: load i32, i32* %local
; CHECK-NEXT: br label %loop
; CHECK: exit:
; CHECK-NEXT: %[[LCSSAPHI:.*]] = phi i32 [ %x2, %loop ]
; CHECK-NEXT: store i32 %[[LCSSAPHI]], i32* %local
; CHECK-NEXT: %ret = load i32, i32* %local
; CHECK-NEXT: ret i32 %ret
entry:
  %local = alloca i32
  call void @capture(i32* %local)  
  br label %loop

loop:
  %j = phi i32 [ 0, %entry ], [ %next, %loop ]
  %throwaway = call i32 @opaque(i32 %j)
  %x = load i32, i32* %local  
  %x2 = call i32 @opaque(i32 %x)
  store i32 %x2, i32* %local
  %next = add i32 %j, 1
  %cond = icmp eq i32 %next, 0
  br i1 %cond, label %exit, label %loop

exit:
  %ret = load i32, i32* %local
  ret i32 %ret
}


; If the store is "guaranteed modulo exceptions", and the load depends on
; control flow, we can only promote if the pointer is otherwise known to be
; dereferenceable
define i32 @test9() {
; CHECK-LABEL: @test9(
; CHECK: entry:
; CHECK-NEXT: %local = alloca
; CHECK-NEXT: call void @capture(i32* %local)
; CHECK-NEXT: load i32, i32* %local
; CHECK-NEXT: br label %loop
; CHECK: exit:
; CHECK-NEXT: %[[LCSSAPHI:.*]] = phi i32 [ %x2, %else ]
; CHECK-NEXT: store i32 %[[LCSSAPHI]], i32* %local
; CHECK-NEXT: %ret = load i32, i32* %local
; CHECK-NEXT: ret i32 %ret
entry:
  %local = alloca i32
  call void @capture(i32* %local)  
  br label %loop

loop:
  %j = phi i32 [ 0, %entry ], [ %next, %else ]  
  %j2 = call i32 @opaque(i32 %j)
  %cmp = icmp eq i32 %j2, 0
  br i1 %cmp, label %if, label %else

if:  
  %x = load i32, i32* %local
  br label %else

else:
  %x2 = phi i32 [ 0, %loop ], [ %x, %if]
  store i32 %x2, i32* %local
  %next = add i32 %j, 1
  %cond = icmp eq i32 %next, 0
  br i1 %cond, label %exit, label %loop

exit:
  %ret = load i32, i32* %local
  ret i32 %ret
}

define i32 @test9bad(i32 %i) {
; CHECK-LABEL: @test9bad(
; CHECK: entry:
; CHECK-NEXT: %local = alloca
; CHECK-NEXT: call void @capture(i32* %local)
; CHECK-NEXT: %notderef = getelementptr
; CHECK-NEXT: br label %loop
; CHECK: if:
; CHECK-NEXT: load i32, i32* %notderef
; CHECK-NEXT: br label %else
; CHECK: exit:
; CHECK-NEXT: %ret = load i32, i32* %notderef
; CHECK-NEXT: ret i32 %ret
entry:
  %local = alloca i32
  call void @capture(i32* %local)  
  %notderef = getelementptr i32, i32* %local, i32 %i
  br label %loop

loop:
  %j = phi i32 [ 0, %entry ], [ %next, %else ]  
  %j2 = call i32 @opaque(i32 %j)
  %cmp = icmp eq i32 %j2, 0
  br i1 %cmp, label %if, label %else

if:  
  %x = load i32, i32* %notderef
  br label %else

else:
  %x2 = phi i32 [ 0, %loop ], [ %x, %if]
  store i32 %x2, i32* %notderef
  %next = add i32 %j, 1
  %cond = icmp eq i32 %next, 0
  br i1 %cond, label %exit, label %loop

exit:
  %ret = load i32, i32* %notderef
  ret i32 %ret
}

define void @test10(i32 %i) {
Entry:
  br label %Loop
; CHECK-LABEL: @test10(
; CHECK: Entry:
; CHECK-NEXT:   load atomic i32, i32* @X unordered, align 4
; CHECK-NEXT:   br label %Loop


Loop:   ; preds = %Loop, %0
  %j = phi i32 [ 0, %Entry ], [ %Next, %Loop ]    ; <i32> [#uses=1]
  %x = load atomic i32, i32* @X unordered, align 4
  %x2 = add i32 %x, 1
  store atomic i32 %x2, i32* @X unordered, align 4
  %Next = add i32 %j, 1
  %cond = icmp eq i32 %Next, 0
  br i1 %cond, label %Out, label %Loop

Out:
  ret void
; CHECK: Out:
; CHECK-NEXT:   %[[LCSSAPHI:.*]] = phi i32 [ %x2
; CHECK-NEXT:   store atomic i32 %[[LCSSAPHI]], i32* @X unordered, align 4
; CHECK-NEXT:   ret void

}

; Early exit is known not to be taken on first iteration and thus doesn't
; effect whether load is known to execute.
define void @test11(i32 %i) {
Entry:
  br label %Loop
; CHECK-LABEL: @test11(
; CHECK: Entry:
; CHECK-NEXT:   load i32, i32* @X
; CHECK-NEXT:   br label %Loop


Loop:   ; preds = %Loop, %0
  %j = phi i32 [ 0, %Entry ], [ %Next, %body ]    ; <i32> [#uses=1]
  %early.test = icmp ult i32 %j, 32
  br i1 %early.test, label %body, label %Early
body:
  %x = load i32, i32* @X   ; <i32> [#uses=1]
  %x2 = add i32 %x, 1   ; <i32> [#uses=1]
  store i32 %x2, i32* @X
  %Next = add i32 %j, 1   ; <i32> [#uses=2]
  %cond = icmp eq i32 %Next, 0    ; <i1> [#uses=1]
  br i1 %cond, label %Out, label %Loop

Early:
; CHECK: Early:
; CHECK-NEXT:   %[[LCSSAPHI:.*]] = phi i32 [ %x2
; CHECK-NEXT:   store i32 %[[LCSSAPHI]], i32* @X
; CHECK-NEXT:   ret void
  ret void
Out:
  ret void
; CHECK: Out:
; CHECK-NEXT:   %[[LCSSAPHI:.*]] = phi i32 [ %x2
; CHECK-NEXT:   store i32 %[[LCSSAPHI]], i32* @X
; CHECK-NEXT:   ret void

}

!0 = !{!4, !4, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!5, !5, i64 0}
!4 = !{!"int", !1}
!5 = !{!"float", !1}
