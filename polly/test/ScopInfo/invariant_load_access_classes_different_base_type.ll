; RUN: opt %loadPolly -polly-print-scops -polly-invariant-load-hoisting=true -disable-output < %s | FileCheck %s
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
; CODEGEN:    %S.load = load i32, ptr @S
; CODEGEN:    store i32 %S.load, ptr %S.a.preload.s2a
; CODEGEN:    %.load = load float, ptr getelementptr inbounds (i32, ptr @S, i64 1)
; CODEGEN:    store float %.load, ptr %S.b.preload.s2a
;
; CODEGEN:  polly.stmt.for.body:
; CODEGEN:    %p_conv = sitofp i32 %S.load to float
; CODEGEN:    %p_add = fadd float %p_conv, %.load
; CODEGEN:    %p_conv1 = fptosi float %p_add to i32

%struct.anon = type { i32, float }

@S = common global %struct.anon zeroinitializer, align 4

define void @f(ptr %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1000
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %S.a = load i32, ptr @S, align 4
  %conv = sitofp i32 %S.a to float
  %S.b = load float, ptr getelementptr inbounds (%struct.anon, ptr @S, i64 0, i32 1), align 4
  %add = fadd float %conv, %S.b
  %conv1 = fptosi float %add to i32
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %conv1, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
