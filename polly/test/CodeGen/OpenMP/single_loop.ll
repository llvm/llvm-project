; RUN: opt %loadNPMPolly -polly-parallel -polly-parallel-force '-passes=print<polly-ast>' -disable-output < %s | FileCheck %s -check-prefix=AST
; RUN: opt %loadNPMPolly -polly-parallel -polly-parallel-force -passes=polly-codegen -S -verify-dom-info < %s | FileCheck %s -check-prefix=IR

; RUN: opt %loadNPMPolly -polly-parallel -polly-parallel-force '-passes=polly-import-jscop,print<polly-ast>' -disable-output < %s | FileCheck %s -check-prefix=AST-STRIDE4
; RUN: opt %loadNPMPolly -polly-parallel -polly-parallel-force '-passes=polly-import-jscop,polly-codegen' -S < %s | FileCheck %s -check-prefix=IR-STRIDE4

; RUN: opt %loadNPMPolly -polly-parallel -polly-parallel-force -passes=polly-codegen -polly-omp-backend=LLVM -polly-scheduling=static -polly-scheduling-chunksize=43 -S -verify-dom-info < %s | FileCheck %s -check-prefix=LIBOMP-IR-STATIC-CHUNKED
; RUN: opt %loadNPMPolly -polly-parallel -polly-parallel-force -passes=polly-codegen -polly-omp-backend=LLVM -polly-scheduling=static -S -verify-dom-info < %s | FileCheck %s -check-prefix=LIBOMP-IR-STATIC
; RUN: opt %loadNPMPolly -polly-parallel -polly-parallel-force -passes=polly-codegen -polly-omp-backend=LLVM -polly-scheduling=dynamic -S -verify-dom-info < %s | FileCheck %s -check-prefix=LIBOMP-IR-DYNAMIC
; RUN: opt %loadNPMPolly -polly-parallel -polly-parallel-force -passes=polly-codegen -polly-omp-backend=LLVM -polly-scheduling=dynamic -polly-scheduling-chunksize=4 -S -verify-dom-info < %s | FileCheck %s -check-prefix=LIBOMP-IR-DYNAMIC-FOUR
; RUN: opt %loadNPMPolly -polly-parallel -polly-parallel-force '-passes=polly-import-jscop,polly-codegen' -polly-omp-backend=LLVM -S < %s | FileCheck %s -check-prefix=LIBOMP-IR-STRIDE4

; This extensive test case tests the creation of the full set of OpenMP calls
; as well as the subfunction creation using a trivial loop as example.
;
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
; IR-NEXT:   call void @GOMP_parallel_loop_runtime_start(ptr @single_parallel_loop_polly_subfn, ptr %polly.par.userContext, i32 0, i64 0, i64 1024, i64 1)
; IR-NEXT:   call void @single_parallel_loop_polly_subfn(ptr %polly.par.userContext)
; IR-NEXT:   call void @GOMP_parallel_end()
; IR-NEXT:   br label %polly.exiting

; IR: define internal void @single_parallel_loop_polly_subfn(ptr %polly.par.userContext) #1
; IR-LABEL: polly.par.setup:
; IR-NEXT:   %polly.par.LBPtr = alloca i64
; IR-NEXT:   %polly.par.UBPtr = alloca i64
; IR:   br label %polly.par.checkNext

; IR-LABEL: polly.par.exit:
; IR-NEXT:   call void @GOMP_loop_end_nowait()
; IR-NEXT:   ret void

; IR-LABEL: polly.par.checkNext:
; IR-NEXT:   %[[parnext:[._a-zA-Z0-9]*]] = call i8 @GOMP_loop_runtime_next(ptr %polly.par.LBPtr, ptr %polly.par.UBPtr)
; IR-NEXT:   %[[cmp:[._a-zA-Z0-9]*]] = icmp ne i8 %[[parnext]], 0
; IR-NEXT:   br i1 %[[cmp]], label %polly.par.loadIVBounds, label %polly.par.exit

; IR-LABEL: polly.par.loadIVBounds:
; IR-NEXT:   %polly.par.LB = load i64, ptr %polly.par.LBPtr
; IR-NEXT:   %polly.par.UB = load i64, ptr %polly.par.UBPtr
; IR-NEXT:   %polly.par.UBAdjusted = sub i64 %polly.par.UB, 1
; IR-NEXT:   br label %polly.loop_preheader

; IR-LABEL: polly.loop_exit:
; IR-NEXT:   br label %polly.par.checkNext

; IR-LABEL: polly.loop_header:
; IR-NEXT:   %polly.indvar = phi i64 [ %polly.par.LB, %polly.loop_preheader ], [ %polly.indvar_next, %polly.stmt.S ]
; IR-NEXT:   br label %polly.stmt.S

; IR-LABEL: polly.stmt.S:
; IR-NEXT:   %[[offset:[0-9]+]] = shl i64 %polly.indvar, 2
; IR-NEXT:   %[[gep:[._a-zA-Z0-9]*]] = getelementptr i8, ptr {{.*}}, i64 %[[offset]]
; IR-NEXT:   store float 1.000000e+00, ptr %[[gep]]
; IR-NEXT:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; IR-NEXT:   %polly.loop_cond = icmp sle i64 %polly.indvar_next, %polly.par.UBAdjusted
; IR-NEXT:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

; IR-LABEL: polly.loop_preheader:
; IR-NEXT:   br label %polly.loop_header

; IR: attributes #1 = { "polly.skip.fn" }

; IR-STRIDE4:   call void @GOMP_parallel_loop_runtime_start(ptr @single_parallel_loop_polly_subfn, ptr %polly.par.userContext, i32 0, i64 0, i64 1024, i64 4)
; IR-STRIDE4:  add nsw i64 %polly.indvar, 3
; IR-STRIDE4:  %polly.indvar_next = add nsw i64 %polly.indvar, 4
; IR-STRIDE4   %polly.adjust_ub = sub i64 %polly.par.UBAdjusted, 4

; LIBOMP-IR-STATIC-CHUNKED: %struct.ident_t = type { i32, i32, i32, i32, ptr }

; LIBOMP-IR-STATIC-CHUNKED-LABEL: single_parallel_loop()
; LIBOMP-IR-STATIC-CHUNKED-NEXT: entry
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.par.userContext = alloca

; LIBOMP-IR-STATIC-CHUNKED-LABEL: polly.parallel.for:
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @.loc.dummy, i32 4, ptr @single_parallel_loop_polly_subfn, i64 0, i64 1024, i64 1, ptr %polly.par.userContext)
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   br label %polly.exiting

; LIBOMP-IR-STATIC-CHUNKED: define internal void @single_parallel_loop_polly_subfn(ptr %polly.kmpc.global_tid, ptr %polly.kmpc.bound_tid, i64 %polly.kmpc.lb, i64 %polly.kmpc.ub, i64 %polly.kmpc.inc, ptr %polly.kmpc.shared)
; LIBOMP-IR-STATIC-CHUNKED-LABEL: polly.par.setup:
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.par.LBPtr = alloca i64
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.par.UBPtr = alloca i64
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.par.lastIterPtr = alloca i32
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.par.StridePtr = alloca i64
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.par.global_tid = load i32, ptr %polly.kmpc.global_tid
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   store i64 %polly.kmpc.lb, ptr %polly.par.LBPtr
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   store i64 %polly.kmpc.ub, ptr %polly.par.UBPtr
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   store i32 0, ptr %polly.par.lastIterPtr
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   store i64 %polly.kmpc.inc, ptr %polly.par.StridePtr
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.indvar.UBAdjusted = add i64 %polly.kmpc.ub, -1
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   store i64 %polly.indvar.UBAdjusted, ptr %polly.par.UBPtr, align 8
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   call void @__kmpc_for_static_init_{{[4|8]}}(ptr @.loc.dummy{{[.0-9]*}}, i32 %polly.par.global_tid, i32 33, ptr %polly.par.lastIterPtr, ptr %polly.par.LBPtr, ptr %polly.par.UBPtr, ptr %polly.par.StridePtr, i64 1, i64 43)
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.kmpc.stride = load i64, ptr %polly.par.StridePtr, align 8
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.indvar.LB = load i64, ptr %polly.par.LBPtr
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.indvar.UB.temp = load i64, ptr %polly.par.UBPtr
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.indvar.UB.inRange = icmp sle i64 %polly.indvar.UB.temp, %polly.indvar.UBAdjusted
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.indvar.UB = select i1 %polly.indvar.UB.inRange, i64 %polly.indvar.UB.temp, i64 %polly.indvar.UBAdjusted
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   store i64 %polly.indvar.UB, ptr %polly.par.UBPtr, align 8
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.hasIteration = icmp sle i64 %polly.indvar.LB, %polly.indvar.UB
; LIBOMP-IR-STATIC-CHUNKED:   br i1 %polly.hasIteration, label %polly.par.loadIVBounds, label %polly.par.exit

; LIBOMP-IR-STATIC-CHUNKED-LABEL: polly.par.exit:
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   call void @__kmpc_for_static_fini(ptr @.loc.dummy, i32 %polly.par.global_tid)
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   ret void

; LIBOMP-IR-STATIC-CHUNKED-LABEL: polly.par.checkNext:
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.indvar.nextLB = add i64 %polly.indvar.LB.entry, %polly.kmpc.stride
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %{{[0-9]+}} = add i64 %polly.indvar.UB.entry, %polly.kmpc.stride
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.indvar.nextUB.outOfBounds = icmp sgt i64 %{{[0-9]+}}, %polly.indvar.UBAdjusted
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.indvar.nextUB = select i1 %polly.indvar.nextUB.outOfBounds, i64 %polly.indvar.UBAdjusted, i64 %{{[0-9]+}}
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   store i64 %polly.indvar.nextLB, ptr %polly.par.LBPtr, align 8
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   store i64 %polly.indvar.nextUB, ptr %polly.par.UBPtr, align 8
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.hasWork = icmp sle i64 %polly.indvar.nextLB, %polly.indvar.UBAdjusted
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   br i1 %polly.hasWork, label %polly.par.loadIVBounds, label %polly.par.exit

; LIBOMP-IR-STATIC-CHUNKED-LABEL: polly.par.loadIVBounds:
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.indvar.LB.entry = load i64, ptr %polly.par.LBPtr, align 8
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.indvar.UB.entry = load i64, ptr %polly.par.UBPtr, align 8
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   br label %polly.loop_preheader

; LIBOMP-IR-STATIC-CHUNKED-LABEL: polly.loop_exit:
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   br label %polly.par.checkNext

; LIBOMP-IR-STATIC-CHUNKED-LABEL: polly.loop_header:
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.indvar = phi i64 [ %polly.indvar.LB.entry, %polly.loop_preheader ], [ %polly.indvar_next, %polly.stmt.S ]
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   br label %polly.stmt.S

; LIBOMP-IR-STATIC-CHUNKED-LABEL: polly.stmt.S:
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %[[offset:[0-9]+]] = shl i64 %polly.indvar, 2
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %[[gep:[._a-zA-Z0-9]*]] = getelementptr i8, ptr {{.*}}, i64 %[[offset]]
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   store float 1.000000e+00, ptr %[[gep]]
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   %polly.loop_cond = icmp sle i64 %polly.indvar_next, %polly.indvar.UB.entry
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

; LIBOMP-IR-STATIC-CHUNKED-LABEL: polly.loop_preheader:
; LIBOMP-IR-STATIC-CHUNKED-NEXT:   br label %polly.loop_header

; LIBOMP-IR-STATIC-CHUNKED: attributes #1 = { "polly.skip.fn" }

; LIBOMP-IR-STATIC: define internal void @single_parallel_loop_polly_subfn(ptr %polly.kmpc.global_tid, ptr %polly.kmpc.bound_tid, i64 %polly.kmpc.lb, i64 %polly.kmpc.ub, i64 %polly.kmpc.inc, ptr %polly.kmpc.shared)
; LIBOMP-IR-STATIC-LABEL: polly.par.setup:
; LIBOMP-IR-STATIC:   call void @__kmpc_for_static_init_{{[4|8]}}(ptr @.loc.dummy{{[.0-9]*}}, i32 %polly.par.global_tid, i32 34, ptr %polly.par.lastIterPtr, ptr %polly.par.LBPtr, ptr %polly.par.UBPtr, ptr %polly.par.StridePtr, i64 1, i64 1)
; LIBOMP-IR-STATIC:   br i1 %polly.hasIteration, label %polly.par.loadIVBounds, label %polly.par.exit

; LIBOMP-IR-STATIC-LABEL: polly.par.exit:
; LIBOMP-IR-STATIC-NEXT:   call void @__kmpc_for_static_fini(ptr @.loc.dummy, i32 %polly.par.global_tid)
; LIBOMP-IR-STATIC-NEXT:   ret void

; LIBOMP-IR-STATIC-LABEL: polly.par.checkNext:
; LIBOMP-IR-STATIC-NEXT:   br label %polly.par.exit

; LIBOMP-IR-STATIC-LABEL: polly.par.loadIVBounds:
; LIBOMP-IR-STATIC-NEXT:   br label %polly.loop_preheader

; LIBOMP-IR-STATIC-LABEL: polly.loop_exit:
; LIBOMP-IR-STATIC-NEXT:   br label %polly.par.checkNext

; LIBOMP-IR-STATIC-LABEL: polly.loop_header:
; LIBOMP-IR-STATIC-NEXT:   %polly.indvar = phi i64 [ %polly.indvar.LB, %polly.loop_preheader ], [ %polly.indvar_next, %polly.stmt.S ]
; LIBOMP-IR-STATIC-NEXT:   br label %polly.stmt.S

; LIBOMP-IR-STATIC-LABEL: polly.stmt.S:
; LIBOMP-IR-STATIC-NEXT:   %[[offset:[0-9]+]] = shl i64 %polly.indvar, 2
; LIBOMP-IR-STATIC-NEXT:   %[[gep:[._a-zA-Z0-9]*]] = getelementptr i8, ptr {{.*}}, i64 %[[offset]]
; LIBOMP-IR-STATIC-NEXT:   store float 1.000000e+00, ptr %[[gep]]
; LIBOMP-IR-STATIC-NEXT:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; LIBOMP-IR-STATIC-NEXT:   %polly.loop_cond = icmp sle i64 %polly.indvar_next, %polly.indvar.UB
; LIBOMP-IR-STATIC-NEXT:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

; LIBOMP-IR-STATIC-LABEL: polly.loop_preheader:
; LIBOMP-IR-STATIC-NEXT:   br label %polly.loop_header

; LIBOMP-IR-DYNAMIC:   call void @__kmpc_dispatch_init_{{[4|8]}}(ptr @.loc.dummy, i32 %polly.par.global_tid, i32 35, i64 %polly.kmpc.lb, i64 %polly.indvar.UBAdjusted, i64 %polly.kmpc.inc, i64 1)
; LIBOMP-IR-DYNAMIC-NEXT:   %{{[0-9]+}} = call i32 @__kmpc_dispatch_next_{{[4|8]}}(ptr @.loc.dummy, i32 %polly.par.global_tid, ptr %polly.par.lastIterPtr, ptr %polly.par.LBPtr, ptr %polly.par.UBPtr, ptr %polly.par.StridePtr)
; LIBOMP-IR-DYNAMIC-NEXT:   %polly.hasIteration = icmp eq i32 %{{[0-9]+}}, 1
; LIBOMP-IR-DYNAMIC-NEXT:   br i1 %polly.hasIteration, label %polly.par.loadIVBounds, label %polly.par.exit

; LIBOMP-IR-DYNAMIC-LABEL: polly.par.exit:
; LIBOMP-IR-DYNAMIC-NEXT:   ret void

; LIBOMP-IR-DYNAMIC-LABEL: polly.par.checkNext:
; LIBOMP-IR-DYNAMIC-NEXT:   %{{[0-9]+}} = call i32 @__kmpc_dispatch_next_{{[4|8]}}(ptr @.loc.dummy, i32 %polly.par.global_tid, ptr %polly.par.lastIterPtr, ptr %polly.par.LBPtr, ptr %polly.par.UBPtr, ptr %polly.par.StridePtr)
; LIBOMP-IR-DYNAMIC-NEXT:   %polly.hasWork = icmp eq i32 %{{[0-9]+}}, 1
; LIBOMP-IR-DYNAMIC-NEXT:   br i1 %polly.hasWork, label %polly.par.loadIVBounds, label %polly.par.exit

; LIBOMP-IR-DYNAMIC-LABEL: polly.par.loadIVBounds:
; LIBOMP-IR-DYNAMIC-NEXT:   %polly.indvar.LB = load i64, ptr %polly.par.LBPtr
; LIBOMP-IR-DYNAMIC-NEXT:   %polly.indvar.UB = load i64, ptr %polly.par.UBPtr
; LIBOMP-IR-DYNAMIC-NEXT:   br label %polly.loop_preheader

; LIBOMP-IR-DYNAMIC-FOUR:   call void @__kmpc_dispatch_init_{{[4|8]}}(ptr @.loc.dummy, i32 %polly.par.global_tid, i32 35, i64 %polly.kmpc.lb, i64 %polly.indvar.UBAdjusted, i64 %polly.kmpc.inc, i64 4)

; LIBOMP-IR-STRIDE4:     call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @.loc.dummy, i32 4, ptr @single_parallel_loop_polly_subfn, i64 0, i64 1024, i64 4, ptr %polly.par.userContext)

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [1024 x float] zeroinitializer, align 16

define void @single_parallel_loop() nounwind {
entry:
  br label %for.i

for.i:
  %indvar = phi i64 [ %indvar.next, %for.inc], [ 0, %entry ]
  %scevgep = getelementptr [1024 x float], ptr @A, i64 0, i64 %indvar
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %S, label %exit

S:
  store float 1.0, ptr %scevgep
  br label %for.inc

for.inc:
  %indvar.next = add i64 %indvar, 1
  br label %for.i

exit:
  ret void
}
