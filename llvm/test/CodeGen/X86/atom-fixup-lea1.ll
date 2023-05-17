; RUN: llc < %s -mcpu=atom -mtriple=i686-linux | FileCheck %s
; CHECK: addl
; CHECK-NEXT:leal
; CHECK-NEXT:decl
; CHECK-NEXT:jne

; Test for the FixupLEAs pre-emit pass. An LEA should be substituted for the ADD
; that increments the array pointer because it is within 5 instructions of the
; corresponding load. The ADD precedes the load by following the loop back edge.

; Original C code
;int test(int n, int * array)
;{
;  int sum = 0;
;  for(int i = 0; i < n; i++)
;    sum += array[i];
;  return sum;
;}

define i32 @test(i32 %n, ptr nocapture %array) {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body, label %for.end

for.body:
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %sum.05 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %array, i32 %i.06
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %sum.05
  %inc = add nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa
}
