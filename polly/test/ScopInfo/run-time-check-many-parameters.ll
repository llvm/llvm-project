; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; A valid Scop would print the list of it's statements, we check that we do not
; see that list.
;
; CHECK-NOT: Statements
;
; FIXME: Handling this is an open problem, at the moment we just bail out.
;
; void foo(float *A, float *B,
; 	long p1,
; 	long p2,
; 	long p3,
; 	long p4,
; 	long p5,
; 	long p6,
; 	long p7,
; 	long p8,
; 	long p9,
; 	long p10,
; 	long p11,
; 	long p12) {
;   for (long i = 0; i < 100; i++) {
;     A[i] =
; 	B[i + p1] +
; 	B[i + p2] +
; 	B[i + p3] +
; 	B[i + p4] +
; 	B[i + p5] +
; 	B[i + p6] +
; 	B[i + p7] +
; 	B[i + p8] +
; 	B[i + p9] +
; 	B[i + p10] +
; 	B[i + p11] +
; 	B[i + p12];
;   }
; }
;
; Computing the minimal and maximal element accessed in B is very expensive.
; Expressing the minimal element itself yields a rather complex isl_pw_aff which
; looks as follows:
; { ...
;   MemRef_B[(100 + p11)] : p2 <= -1 + p1 and p3 <= -1 + p1 and p4 <= -1 + p1
;                           and p5 <= -1 + p1 and p6 <= -1 + p1 and
;                           p7 <= -1 + p1 and p8 <= -1 + p1 and p9 <= -1 + p1
;                           and p10 <= -1 + p1 and p11 >= p1 and
;                           p12 <= -1 + p11;
;   MemRef_B[(100 + p12)] : p2 <= -1 + p1 and p3 <= -1 + p1 and p4 <= -1 + p1
;                           and p5 <= -1 + p1 and p6 <= -1 + p1 and
;                           p7 <= -1 + p1 and p8 <= -1 + p1 and p9 <= -1 + p1
;                           and p10 <= -1 + p1 and p11 <= -1 + p1 and p12 >= p1;
;
; and this isl_pw_aff is then 1:1 translated into a isl ast expression.
;
; In the best case, we would create a run-time check such as:
;
; if (B[99 + max(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)] < A[0]
;     || A[99] B[min(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]))
;

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @foo(ptr %A, ptr %B, i64 %p1, i64 %p2, i64 %p3, i64 %p4, i64 %p5, i64 %p6, i64 %p7, i64 %p8, i64 %p9, i64 %p10, i64 %p11, i64 %p12) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.body
  %i.01 = phi i64 [ 0, %entry.split ], [ %tmp25, %for.body ]
  %tmp = add i64 %p1, %i.01
  %arrayidx = getelementptr float, ptr %B, i64 %tmp
  %tmp2 = add i64 %p2, %i.01
  %arrayidx2 = getelementptr float, ptr %B, i64 %tmp2
  %tmp3 = add i64 %p3, %i.01
  %arrayidx5 = getelementptr float, ptr %B, i64 %tmp3
  %tmp4 = add i64 %p4, %i.01
  %arrayidx8 = getelementptr float, ptr %B, i64 %tmp4
  %tmp5 = add i64 %p5, %i.01
  %arrayidx11 = getelementptr float, ptr %B, i64 %tmp5
  %tmp6 = add i64 %p6, %i.01
  %arrayidx14 = getelementptr float, ptr %B, i64 %tmp6
  %tmp7 = add i64 %p7, %i.01
  %arrayidx17 = getelementptr float, ptr %B, i64 %tmp7
  %tmp8 = add i64 %p8, %i.01
  %arrayidx20 = getelementptr float, ptr %B, i64 %tmp8
  %tmp9 = add i64 %p9, %i.01
  %arrayidx23 = getelementptr float, ptr %B, i64 %tmp9
  %tmp10 = add i64 %p10, %i.01
  %arrayidx26 = getelementptr float, ptr %B, i64 %tmp10
  %tmp11 = add i64 %p11, %i.01
  %arrayidx29 = getelementptr float, ptr %B, i64 %tmp11
  %tmp12 = add i64 %p12, %i.01
  %arrayidx32 = getelementptr float, ptr %B, i64 %tmp12
  %arrayidx34 = getelementptr float, ptr %A, i64 %i.01
  %tmp13 = load float, ptr %arrayidx, align 4
  %tmp14 = load float, ptr %arrayidx2, align 4
  %add3 = fadd float %tmp13, %tmp14
  %tmp15 = load float, ptr %arrayidx5, align 4
  %add6 = fadd float %add3, %tmp15
  %tmp16 = load float, ptr %arrayidx8, align 4
  %add9 = fadd float %add6, %tmp16
  %tmp17 = load float, ptr %arrayidx11, align 4
  %add12 = fadd float %add9, %tmp17
  %tmp18 = load float, ptr %arrayidx14, align 4
  %add15 = fadd float %add12, %tmp18
  %tmp19 = load float, ptr %arrayidx17, align 4
  %add18 = fadd float %add15, %tmp19
  %tmp20 = load float, ptr %arrayidx20, align 4
  %add21 = fadd float %add18, %tmp20
  %tmp21 = load float, ptr %arrayidx23, align 4
  %add24 = fadd float %add21, %tmp21
  %tmp22 = load float, ptr %arrayidx26, align 4
  %add27 = fadd float %add24, %tmp22
  %tmp23 = load float, ptr %arrayidx29, align 4
  %add30 = fadd float %add27, %tmp23
  %tmp24 = load float, ptr %arrayidx32, align 4
  %add33 = fadd float %add30, %tmp24
  store float %add33, ptr %arrayidx34, align 4
  %tmp25 = add nsw i64 %i.01, 1
  %exitcond = icmp ne i64 %tmp25, 100
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}
