; Test that the reduction dependences are always initialised, even in a case
; where we have no reduction. If this object is NULL, then isl operations on
; it will fail.
; RUN: opt -S %loadPolly -polly-dependences -analyze < %s | FileCheck %s -check-prefix=VALUE
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

;     for(i = 0; i < 100; i++ )
; S1:   A[i] = 2;

define void @sequential_writes() {
entry:
  %A = alloca [200 x i32]
  br label %S1

S1:
  %indvar.1 = phi i64 [ 0, %entry ], [ %indvar.next.1, %S1 ]
  %arrayidx.1 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.1
  store i32 2, i32* %arrayidx.1
  %indvar.next.1 = add i64 %indvar.1, 1
  %exitcond.1 = icmp ne i64 %indvar.next.1, 100
  br i1 %exitcond.1, label %S1, label %exit.1

exit.1:
  ret void
}

; VALUE:      RAW dependences:
; VALUE-NEXT:     {  }
; VALUE-NEXT: WAR dependences:
; VALUE-NEXT:     {  }
; VALUE-NEXT: WAW dependences:
; VALUE-NEXT:     {  }
; VALUE-NEXT: Reduction dependences:
; VALUE-NEXT:     {  }
