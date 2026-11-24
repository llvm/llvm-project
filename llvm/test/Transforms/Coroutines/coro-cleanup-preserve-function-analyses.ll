; RUN: opt -disable-output -debug-pass-manager -passes='function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>),coro-cleanup,function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s

; Lowering @lowered changes its CFG, but should not invalidate the analyses of
; @other.

; CHECK: Running pass: CoroCleanupPass on [module]{{$}}
; CHECK-NEXT: Invalidating analysis: DominatorTreeAnalysis on lowered{{$}}
; CHECK-NEXT: Invalidating analysis: LazyValueAnalysis on lowered{{$}}
; CHECK-NEXT: Running pass: SimplifyCFGPass on lowered (10 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on lowered (10 instructions){{$}}
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on lowered{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on lowered (10 instructions){{$}}
; CHECK-NEXT: Running analysis: LazyValueAnalysis on lowered{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on lowered (10 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on lowered (10 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

define void @lowered(i1 %c) personality ptr @__gxx_personality_v0 {
entry:
  %hdl = call ptr @llvm.coro.noop()
  %fn = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  br i1 %c, label %a, label %b

a:
  call void @g()
  invoke fastcc void %fn(ptr %hdl)
          to label %cont unwind label %lpad

b:
  invoke void @g()
          to label %cont unwind label %lpad

cont:
  %p = phi i32 [ 1, %a ], [ 2, %b ]
  call void @use(i32 %p)
  ret void

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  call void @use(i32 3)
  resume { ptr, i32 } %lp
}

define void @other() {
  ret void
}

declare void @g()
declare void @use(i32)
declare ptr @llvm.coro.noop()
declare ptr @llvm.coro.subfn.addr(ptr, i8)
declare i32 @__gxx_personality_v0(...)
