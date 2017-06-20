; RUN: llc < %s -mcpu=atom -mtriple=i686-linux | FileCheck %s
; CHECK: addl ([[reg:%[a-z]+]])
; CHECK-NEXT: addl $4, [[reg]]

; Test for the FixupLEAs pre-emit pass.
; An LEA should NOT be substituted for the ADD instruction
; that increments the array pointer if it is greater than 5 instructions
; away from the memory reference that uses it.

; Original C code: clang -m32 -S -O2
;int test(int n, int * array, int * m, int * array2)
;{
;  int i, j = 0;
;  int sum = 0;
;  for (i = 0, j = 0; i < n;) {
;    ++i;
;    *m += array2[j++];
;    sum += array[i];
;  }
;  return sum;
;}

define i32 @test(i32 %n, i32* nocapture %array, i32* nocapture %m, i32* nocapture %array2) #0 {
entry:
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %.pre = load i32, i32* %m, align 4
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %0 = phi i32 [ %.pre, %for.body.lr.ph ], [ %add, %for.body ]
  %sum.010 = phi i32 [ 0, %for.body.lr.ph ], [ %add3, %for.body ]
  %j.09 = phi i32 [ 0, %for.body.lr.ph ], [ %inc1, %for.body ]
  %inc1 = add nsw i32 %j.09, 1
  %arrayidx = getelementptr inbounds i32, i32* %array2, i32 %j.09
  %1 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %1
  store i32 %add, i32* %m, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %array, i32 %inc1
  %2 = load i32, i32* %arrayidx2, align 4
  %add3 = add nsw i32 %2, %sum.010
  %exitcond = icmp eq i32 %inc1, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  ret i32 %sum.0.lcssa
}

