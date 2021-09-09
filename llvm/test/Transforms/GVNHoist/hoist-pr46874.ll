; RUN: opt -gvn-hoist -S < %s | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@input = local_unnamed_addr global i32* null, align 8

; Check that the load instruction is **not** hoisted
; CHECK-LABEL: @_Z3fooPii
; CHECK-LABEL: if.then:
; CHECK-NEXT: load
; CHECK-LABEL: if2:
; CHECK: load
; CHECK-LABEL: @main

define i32 @_Z3fooPii(i32* %p, i32 %x) local_unnamed_addr  {
entry:
  %cmp.not = icmp eq i32* %p, null
  br i1 %cmp.not, label %if.end3, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i32, i32* %p, align 4, !tbaa !3
  %add = add nsw i32 %0, %x
  %cmp1 = icmp eq i32 %add, 4
  br i1 %cmp1, label %if2, label %if.end3

if.end3:                                          ; preds = %entry, %if.then
  %x.addr.0 = phi i32 [ %add, %if.then ], [ %x, %entry ]
  %add4 = add nsw i32 %x.addr.0, 2
  br i1 %cmp.not, label %if.end11, label %if2

if2:                                              ; preds = %if.end3, %if.then
  %x.addr.1 = phi i32 [ 4, %if.then ], [ %x.addr.0, %if.end3 ]
  %y.0 = phi i32 [ 2, %if.then ], [ %add4, %if.end3 ]
  %1 = load i32, i32* %p, align 4, !tbaa !3
  %add7 = add nsw i32 %x.addr.1, %1
  %cmp8 = icmp eq i32 %add7, 5
  br i1 %cmp8, label %end, label %if.end11

if.end11:                                         ; preds = %if.end3, %if2
  %x.addr.2 = phi i32 [ %add7, %if2 ], [ %x.addr.0, %if.end3 ]
  %y.1 = phi i32 [ %y.0, %if2 ], [ %add4, %if.end3 ]
  %add12 = add nsw i32 %y.1, %x.addr.2
  br label %end

end:                                              ; preds = %if2, %if.end11
  %x.addr.3 = phi i32 [ 5, %if2 ], [ %x.addr.2, %if.end11 ]
  %y.2 = phi i32 [ %y.0, %if2 ], [ %add12, %if.end11 ]
  %add13 = add nsw i32 %x.addr.3, %y.2
  ret i32 %add13
}

define i32 @main() local_unnamed_addr  {
entry:
  %0 = load i32*, i32** @input, align 8, !tbaa !7
  %call = call i32 @_Z3fooPii(i32* %0, i32 0)
  ret i32 %call
}


!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"pointer@_ZTSPi", !5, i64 0}
