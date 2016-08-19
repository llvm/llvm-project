; RUN: opt %loadPolly -polly-scops -polly-invariant-load-hoisting=true -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -polly-invariant-load-hoisting=true -S < %s | FileCheck %s --check-prefix=CODEGEN
;
;    struct {
;      int a;
;      float b;
;    } S;
;
;    void f(int *A) {
;      for (int i = 0; i < 1000; i++)
;        A[i] = S.a + S.b;
;    }
;
; CHECK:    Invariant Accesses: {
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_for_body[i0] -> MemRef_S[0] };
; CHECK:            Execution Context: {  :  }
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_for_body[i0] -> MemRef_S[1] };
; CHECK:            Execution Context: {  :  }
; CHECK:    }
;
; CODEGEN:    %S.b.preload.s2a = alloca float
; CODEGEN:    %S.a.preload.s2a = alloca i32
;
; CODEGEN:    %.load = load i32, i32* getelementptr inbounds (%struct.anon, %struct.anon* @S, i32 0, i32 0)
; CODEGEN:    store i32 %.load, i32* %S.a.preload.s2a
; CODEGEN:    %.load1 = load float, float* bitcast (i32* getelementptr (i32, i32* getelementptr inbounds (%struct.anon, %struct.anon* @S, i32 0, i32 0), i64 1) to float*)
; CODEGEN:    store float %.load1, float* %S.b.preload.s2a
;
; CODEGEN:  polly.stmt.for.body:
; CODEGEN:    %p_conv = sitofp i32 %.load to float
; CODEGEN:    %p_add = fadd float %p_conv, %.load1
; CODEGEN:    %p_conv1 = fptosi float %p_add to i32

%struct.anon = type { i32, float }

@S = common global %struct.anon zeroinitializer, align 4

define void @f(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1000
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %S.a = load i32, i32* getelementptr inbounds (%struct.anon, %struct.anon* @S, i64 0, i32 0), align 4
  %conv = sitofp i32 %S.a to float
  %S.b = load float, float* getelementptr inbounds (%struct.anon, %struct.anon* @S, i64 0, i32 1), align 4
  %add = fadd float %conv, %S.b
  %conv1 = fptosi float %add to i32
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %conv1, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
