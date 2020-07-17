; RUN: opt -analyze -enable-new-pm=0 -scalar-evolution < %s | FileCheck %s
; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

@arr1 = internal unnamed_addr constant [50 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50], align 4
@arr2 = internal unnamed_addr constant [50 x i32] [i32 49, i32 48, i32 47, i32 46, i32 45, i32 44, i32 43, i32 42, i32 41, i32 40, i32 39, i32 38, i32 37, i32 36, i32 35, i32 34, i32 33, i32 32, i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0], align 4

; PR11034
define i32 @test1() nounwind readnone {
; CHECK: test1
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %sum.04 = phi i32 [ 0, %entry ], [ %add2, %for.body ]
; CHECK: -->  %sum.04{{ U: [^ ]+ S: [^ ]+}}{{ *}}Exits: 2450
  %i.03 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [50 x i32], [50 x i32]* @arr1, i32 0, i32 %i.03
  %0 = load i32, i32* %arrayidx, align 4
; CHECK: -->  %0{{ U: [^ ]+ S: [^ ]+}}{{ *}}Exits: 50
  %arrayidx1 = getelementptr inbounds [50 x i32], [50 x i32]* @arr2, i32 0, i32 %i.03
  %1 = load i32, i32* %arrayidx1, align 4
; CHECK: -->  %1{{ U: [^ ]+ S: [^ ]+}}{{ *}}Exits: 0
  %add = add i32 %0, %sum.04
  %add2 = add i32 %add, %1
  %inc = add nsw i32 %i.03, 1
  %cmp = icmp eq i32 %inc, 50
  br i1 %cmp, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 %add2
}


%struct.ListNode = type { %struct.ListNode*, i32 }

@node5 = internal constant { %struct.ListNode*, i32, [4 x i8] } { %struct.ListNode* bitcast ({ %struct.ListNode*, i32, [4 x i8] }* @node4 to %struct.ListNode*), i32 4, [4 x i8] undef }, align 8
@node4 = internal constant { %struct.ListNode*, i32, [4 x i8] } { %struct.ListNode* bitcast ({ %struct.ListNode*, i32, [4 x i8] }* @node3 to %struct.ListNode*), i32 3, [4 x i8] undef }, align 8
@node3 = internal constant { %struct.ListNode*, i32, [4 x i8] } { %struct.ListNode* bitcast ({ %struct.ListNode*, i32, [4 x i8] }* @node2 to %struct.ListNode*), i32 2, [4 x i8] undef }, align 8
@node2 = internal constant { %struct.ListNode*, i32, [4 x i8] } { %struct.ListNode* bitcast ({ %struct.ListNode*, i32, [4 x i8] }* @node1 to %struct.ListNode*), i32 1, [4 x i8] undef }, align 8
@node1 = internal constant { %struct.ListNode*, i32, [4 x i8] } { %struct.ListNode* null, i32 0, [4 x i8] undef }, align 8

define i32 @test2() nounwind uwtable readonly {
; CHECK: test2
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %sum.02 = phi i32 [ 0, %entry ], [ %add, %for.body ]
; CHECK: -->  %sum.02{{ U: [^ ]+ S: [^ ]+}}{{ *}}Exits: 10
  %n.01 = phi %struct.ListNode* [ bitcast ({ %struct.ListNode*, i32, [4 x i8] }* @node5 to %struct.ListNode*), %entry ], [ %1, %for.body ]
; CHECK: -->  %n.01{{ U: [^ ]+ S: [^ ]+}}{{ *}}Exits: @node1
  %i = getelementptr inbounds %struct.ListNode, %struct.ListNode* %n.01, i64 0, i32 1
  %0 = load i32, i32* %i, align 4
  %add = add nsw i32 %0, %sum.02
  %next = getelementptr inbounds %struct.ListNode, %struct.ListNode* %n.01, i64 0, i32 0
  %1 = load %struct.ListNode*, %struct.ListNode** %next, align 8
; CHECK: -->  %1{{ U: [^ ]+ S: [^ ]+}}{{ *}}Exits: 0
  %cmp = icmp eq %struct.ListNode* %1, null
  br i1 %cmp, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 %add
}
