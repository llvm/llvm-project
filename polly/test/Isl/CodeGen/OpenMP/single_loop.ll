; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-ast -analyze < %s | FileCheck %s -check-prefix=AST
; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-codegen -S -verify-dom-info < %s | FileCheck %s -check-prefix=IR

; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-import-jscop -polly-import-jscop-dir=%S -polly-ast -analyze < %s | FileCheck %s -check-prefix=AST-STRIDE4
; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-import-jscop -polly-import-jscop-dir=%S -polly-codegen -S < %s | FileCheck %s -check-prefix=IR-STRIDE4

; This extensive test case tests the creation of the full set of OpenMP calls
; as well as the subfunction creation using a trivial loop as example.

; #define N 1024
; float A[N];
;
; void single_parallel_loop(void) {
;   for (long i = 0; i < N; i++)
;     A[i] = 1;
; }

; AST: #pragma simd
; AST: #pragma omp parallel for
; AST: for (int c0 = 0; c0 <= 1023; c0 += 1)
; AST:   Stmt_S(c0);

; AST-STRIDE4: #pragma omp parallel for
; AST-STRIDE4: for (int c0 = 0; c0 <= 1023; c0 += 4)
; AST-STRIDE4:   #pragma simd
; AST-STRIDE4:   for (int c1 = c0; c1 <= c0 + 3; c1 += 1)
; AST-STRIDE4:     Stmt_S(c1);

; IR-LABEL: single_parallel_loop()
; IR-NEXT: entry
; IR-NEXT:   %polly.par.userContext = alloca

; IR-LABEL: polly.parallel.for:
; IR-NEXT:   %0 = bitcast {}* %polly.par.userContext to i8*
; IR-NEXT:   call void @llvm.lifetime.start(i64 0, i8* %0)
; IR-NEXT:   %polly.par.userContext1 = bitcast {}* %polly.par.userContext to i8*
; IR-NEXT:   call void @GOMP_parallel_loop_runtime_start(void (i8*)* @single_parallel_loop_polly_subfn, i8* %polly.par.userContext1, i32 0, i64 0, i64 1024, i64 1)
; IR-NEXT:   call void @single_parallel_loop_polly_subfn(i8* %polly.par.userContext1)
; IR-NEXT:   call void @GOMP_parallel_end()
; IR-NEXT:   %1 = bitcast {}* %polly.par.userContext to i8*
; IR-NEXT:   call void @llvm.lifetime.end(i64 8, i8* %1)
; IR-NEXT:   br label %polly.exiting

; IR: define internal void @single_parallel_loop_polly_subfn(i8* %polly.par.userContext) #2
; IR-LABEL: polly.par.setup:
; IR-NEXT:   %polly.par.LBPtr = alloca i64
; IR-NEXT:   %polly.par.UBPtr = alloca i64
; IR-NEXT:   %polly.par.userContext1 =
; IR:   br label %polly.par.checkNext

; IR-LABEL: polly.par.exit:
; IR-NEXT:   call void @GOMP_loop_end_nowait()
; IR-NEXT:   ret void

; IR-LABEL: polly.par.checkNext:
; IR-NEXT:   %[[parnext:[._a-zA-Z0-9]*]] = call i8 @GOMP_loop_runtime_next(i64* %polly.par.LBPtr, i64* %polly.par.UBPtr)
; IR-NEXT:   %[[cmp:[._a-zA-Z0-9]*]] = icmp ne i8 %[[parnext]], 0
; IR-NEXT:   br i1 %[[cmp]], label %polly.par.loadIVBounds, label %polly.par.exit

; IR-LABEL: polly.par.loadIVBounds:
; IR-NEXT:   %polly.par.LB = load i64, i64* %polly.par.LBPtr
; IR-NEXT:   %polly.par.UB = load i64, i64* %polly.par.UBPtr
; IR-NEXT:   %polly.par.UBAdjusted = sub i64 %polly.par.UB, 1
; IR-NEXT:   br label %polly.loop_preheader

; IR-LABEL: polly.loop_exit:
; IR-NEXT:   br label %polly.par.checkNext

; IR-LABEL: polly.loop_header:
; IR-NEXT:   %polly.indvar = phi i64 [ %polly.par.LB, %polly.loop_preheader ], [ %polly.indvar_next, %polly.stmt.S ]
; IR-NEXT:   br label %polly.stmt.S

; IR-LABEL: polly.stmt.S:
; IR-NEXT:   %[[gep:[._a-zA-Z0-9]*]] = getelementptr [1024 x float], [1024 x float]* {{.*}}, i64 0, i64 %polly.indvar
; IR-NEXT:   store float 1.000000e+00, float* %[[gep]]
; IR-NEXT:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; IR-NEXT:   %polly.adjust_ub = sub i64 %polly.par.UBAdjusted, 1
; IR-NEXT:   %polly.loop_cond = icmp sle i64 %polly.indvar, %polly.adjust_ub
; IR-NEXT:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

; IR-LABEL: polly.loop_preheader:
; IR-NEXT:   br label %polly.loop_header

; IR: attributes #2 = { "polly.skip.fn" }

; IR-STRIDE4:   call void @GOMP_parallel_loop_runtime_start(void (i8*)* @single_parallel_loop_polly_subfn, i8* %polly.par.userContext1, i32 0, i64 0, i64 1024, i64 4)
; IR-STRIDE4:  add nsw i64 %polly.indvar, 3
; IR-STRIDE4:  %polly.indvar_next = add nsw i64 %polly.indvar, 4
; IR-STRIDE4   %polly.adjust_ub = sub i64 %polly.par.UBAdjusted, 4

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [1024 x float] zeroinitializer, align 16

define void @single_parallel_loop() nounwind {
entry:
  br label %for.i

for.i:
  %indvar = phi i64 [ %indvar.next, %for.inc], [ 0, %entry ]
  %scevgep = getelementptr [1024 x float], [1024 x float]* @A, i64 0, i64 %indvar
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %S, label %exit

S:
  store float 1.0, float* %scevgep
  br label %for.inc

for.inc:
  %indvar.next = add i64 %indvar, 1
  br label %for.i

exit:
  ret void
}
