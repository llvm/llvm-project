; RUN: opt %loadPolly -polly-ast -disable-output < %s

;int bar1();
;int bar2();
;int bar3();
;int k;
;#define N 100
;int A[N];
;
;int main() {
;  int i, j, z;
;
;  __sync_synchronize();
;  for (i = 0; i < N; i++) {
;    if (i < 50)
;      A[i] = 8;
;    if (i < 4)
;      A[i] = 9;
;    if (i < 3)
;      A[i] = 10;
;  }
;  __sync_synchronize();
;
;  return A[z];
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [100 x i32] zeroinitializer, align 16 ; <[100 x i32]*> [#uses=2]
@k = common global i32 0, align 4                 ; <i32*> [#uses=0]

define i32 @main() nounwind {
; <label>:0
  fence seq_cst
  br label %1

; <label>:1                                       ; preds = %12, %0
  %indvar = phi i64 [ %indvar.next, %12 ], [ 0, %0 ] ; <i64> [#uses=4]
  %scevgep = getelementptr [100 x i32], [100 x i32]* @A, i64 0, i64 %indvar ; <i32*> [#uses=3]
  %i.0 = trunc i64 %indvar to i32                 ; <i32> [#uses=3]
  %exitcond = icmp ne i64 %indvar, 100            ; <i1> [#uses=1]
  br i1 %exitcond, label %2, label %13

; <label>:2                                       ; preds = %1
  %3 = icmp slt i32 %i.0, 50                      ; <i1> [#uses=1]
  br i1 %3, label %4, label %5

; <label>:4                                       ; preds = %2
  store i32 8, i32* %scevgep
  br label %5

; <label>:5                                       ; preds = %4, %2
  %6 = icmp slt i32 %i.0, 4                       ; <i1> [#uses=1]
  br i1 %6, label %7, label %8

; <label>:7                                       ; preds = %5
  store i32 9, i32* %scevgep
  br label %8

; <label>:8                                       ; preds = %7, %5
  %9 = icmp slt i32 %i.0, 3                       ; <i1> [#uses=1]
  br i1 %9, label %10, label %11

; <label>:10                                      ; preds = %8
  store i32 10, i32* %scevgep
  br label %11

; <label>:11                                      ; preds = %10, %8
  br label %12

; <label>:12                                      ; preds = %11
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %1

; <label>:13                                      ; preds = %1
  fence seq_cst
  %14 = sext i32 undef to i64                     ; <i64> [#uses=1]
  %15 = getelementptr inbounds i32, i32* getelementptr inbounds ([100 x i32], [100 x i32]* @A, i32 0, i32 0), i64 %14 ; <i32*> [#uses=1]
  %16 = load i32, i32* %15                             ; <i32> [#uses=1]
  ret i32 %16
}

; CHECK: for (c2=0;c2<=2;c2++) {
; CHECK:     S0(c2);
; CHECK:       S1(c2);
; CHECK:         S2(c2);
; CHECK: }
; CHECK: S0(3);
; CHECK: S1(3);
; CHECK: for (c2=4;c2<=49;c2++) {
; CHECK:     S0(c2);
; CHECK: }
; CHECK: S0: Stmt_4
; CHECK: S1: Stmt_7
; CHECK: S2: Stmt_10
