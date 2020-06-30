; RUN: opt -S %loadPolly -basic-aa -polly-dependences-analysis-type=value-based -polly-ast -analyze < %s | FileCheck %s
; RUN: opt -S %loadPolly -basic-aa -polly-dependences-analysis-type=value-based -polly-dce -polly-ast -analyze < %s | FileCheck %s -check-prefix=CHECK-DCE
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
;
; for(i = 0; i < 200; i++ )
;   A[i] = 2;
;
; for (i = 0; i < 200; i++ )
;   A[i]  = 5;
;
; for (i = 0; i < 200; i++ )
;   A[i] = 5;
define void @main() nounwind uwtable {
entry:
  %A = alloca [200 x i32], align 16
  br label %for.body.1

for.body.1:
  %indvar.1 = phi i64 [ 0, %entry ], [ %indvar.next.1, %for.body.1 ]
  %arrayidx.1 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.1
  store i32 2, i32* %arrayidx.1, align 4
  %indvar.next.1 = add i64 %indvar.1, 1
  %exitcond.1 = icmp ne i64 %indvar.next.1, 200
  br i1 %exitcond.1, label %for.body.1, label %exit.1

exit.1:
  br label %for.body.2

for.body.2:
  %indvar.2 = phi i64 [ 0, %exit.1 ], [ %indvar.next.2, %for.body.2 ]
  %arrayidx.2 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.2
  store i32 5, i32* %arrayidx.2, align 4
  %indvar.next.2 = add i64 %indvar.2, 1
  %exitcond.2 = icmp ne i64 %indvar.next.2, 200
  br i1 %exitcond.2, label %for.body.2, label %exit.2

exit.2:
  br label %for.body.3

for.body.3:
  %indvar.3 = phi i64 [ 0, %exit.2 ], [ %indvar.next.3, %for.body.3 ]
  %arrayidx.3 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.3
  store i32 7, i32* %arrayidx.3, align 4
  %indvar.next.3 = add i64 %indvar.3, 1
  %exitcond.3 = icmp ne i64 %indvar.next.3, 200
  br i1 %exitcond.3, label %for.body.3 , label %exit.3

exit.3:
  ret void
}

; CHECK: for (int c0 = 0; c0 <= 199; c0 += 1)
; CHECK:   Stmt_for_body_1(c0);
; CHECK: for (int c0 = 0; c0 <= 199; c0 += 1)
; CHECK:   Stmt_for_body_2(c0);
; CHECK: for (int c0 = 0; c0 <= 199; c0 += 1)
; CHECK:   Stmt_for_body_3(c0);

; CHECK-DCE: for (int c0 = 0; c0 <= 199; c0 += 1)
; CHECK-DCE:   Stmt_for_body_3(c0);

