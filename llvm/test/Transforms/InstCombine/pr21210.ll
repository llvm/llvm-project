; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes=instcombine -S | FileCheck %s
; Checks that the select-icmp optimization is safe in two cases
declare void @foo(i32)
declare i32 @bar(i32)

; don't replace 'cond' by 'len' in the home block ('bb') that
; contains the select
define void @test1(i32 %len) {
entry:
  br label %bb

bb:
  %cmp = icmp ult i32 %len, 8
  %cond = select i1 %cmp, i32 %len, i32 8
  call void @foo(i32 %cond)
  %cmp11 = icmp eq i32 %cond, 8
  br i1 %cmp11, label %for.end, label %bb

for.end:
  ret void
; CHECK: select
; CHECK: icmp eq i32 %cond, 8
}

; don't replace 'cond' by 'len' in a block ('b1') that dominates all uses
; of the select outside the home block ('bb'), but can be reached from the home
; block on another path ('bb -> b0 -> b1')
define void @test2(i32 %len) {
entry:
  %0 = call i32 @bar(i32 %len);
  %cmp = icmp ult i32 %len, 4
  br i1 %cmp, label %bb, label %b1
bb:
  %cmp2 = icmp ult i32 %0, 2
  %cond = select i1 %cmp2, i32 %len, i32 8
  %cmp3 = icmp eq i32 %cond, 8
  br i1 %cmp3, label %b0, label %b1

b0:
  call void @foo(i32 %len)
  br label %b1

b1:
; CHECK: phi i32 [ %cond, %bb ], [ undef, %b0 ], [ %0, %entry ]
  %1 = phi i32 [ %cond, %bb ], [ undef, %b0 ], [ %0, %entry ]
  br label %ret

ret:
  call void @foo(i32 %1)
  ret void
}
