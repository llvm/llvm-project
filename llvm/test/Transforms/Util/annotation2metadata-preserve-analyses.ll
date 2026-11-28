; RUN: opt -verify-analysis-invalidation=0 -disable-output -debug-pass-manager -pass-remarks-analysis=annotation-remarks -passes='function(require<domtree>,require<lazy-value-info>),annotation2metadata,function(require<domtree>,require<lazy-value-info>)' %s 2>&1 | FileCheck %s

; Adding !annotation metadata to @f should not invalidate any function
; analyses.

; CHECK: Running pass: Annotation2MetadataPass on [module]{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

@.str = private unnamed_addr constant [8 x i8] c"remarks\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [6 x i8] c"ann.c\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32 }] [
    { ptr, ptr, ptr, i32 } { ptr @f, ptr @.str, ptr @.str.1, i32 1 }
    ], section "llvm.metadata"

define void @f() {
  ret void
}

define void @other() {
  ret void
}
