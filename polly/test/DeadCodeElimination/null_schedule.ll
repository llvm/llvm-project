; RUN: opt -S %loadPolly -basic-aa -polly-dependences-analysis-type=value-based -polly-dce -polly-print-ast -disable-output < %s | FileCheck %s -check-prefix=CHECK-DCE
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
; A[0] = 1;
;
; for(i = 0; i < 100; i++ )
;   A[i+1] = A[i] * 2;
;
; for (i = 0; i < 200; i++ )
;   A[i] = B[i] * 2;

define void @main() nounwind uwtable {

entry:
  %A = alloca [200 x i32], align 16
  %B = alloca [200 x i32], align 16

  %A.zero = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 0
  store i32 1, i32* %A.zero, align 4

  br label %for.body.1

for.body.1:
  %indvar.1 = phi i64 [ 0, %entry ], [ %indvar.next.1, %for.body.1 ]
  %indvar.next.1 = add i64 %indvar.1, 1

  %A.current.1 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.1
  %val1.1 = load i32, i32* %A.current.1, align 4
  %val2.1 = mul i32 %val1.1, 2
  %A.next.1 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.next.1
  store i32 %val2.1, i32* %A.next.1, align 4

  %exitcond.1 = icmp ne i64 %indvar.next.1, 100
  br i1 %exitcond.1, label %for.body.1, label %exit.1

exit.1:
  br label %for.body.2

for.body.2:
  %indvar.2 = phi i64 [ 0, %exit.1 ], [ %indvar.next.2, %for.body.2 ]

  %B.current.2 = getelementptr [200 x i32], [200 x i32]* %B, i64 0, i64 %indvar.2
  %val1.2 = load i32, i32* %B.current.2, align 4
  %val2.2 = mul i32 %val1.2, 2
  %A.current.2 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.2
  store i32 %val2.2, i32* %A.current.2, align 4

  %indvar.next.2 = add i64 %indvar.2, 1
  %exitcond.2 = icmp ne i64 %indvar.next.2, 200
  br i1 %exitcond.2, label %for.body.2, label %exit.3

exit.3:
  ret void
}

; CHECK-DCE: for (int c0 = 0; c0 <= 199; c0 += 1)
; CHECK-DCE:   Stmt_for_body_2(c0);
