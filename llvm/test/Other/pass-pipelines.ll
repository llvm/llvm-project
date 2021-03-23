; Test the particular pass pipelines have the expected structure. This is
; particularly important in order to check that the implicit scheduling of the
; legacy pass manager doesn't introduce unexpected structural changes in the
; pass pipeline.
;
; RUN: opt -enable-new-pm=0 -disable-output -disable-verify -debug-pass=Structure \
; RUN:     -O2 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O2
; RUN: llvm-profdata merge %S/Inputs/pass-pipelines.proftext -o %t.profdata
; RUN: opt -enable-new-pm=0 -disable-output -disable-verify -debug-pass=Structure \
; RUN:     -pgo-kind=pgo-instr-use-pipeline -profile-file='%t.profdata' \
; RUN:     -O2 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O2 --check-prefix=PGOUSE
; RUN: opt -enable-new-pm=0 -disable-output -disable-verify -debug-pass=Structure \
; RUN:     -pgo-kind=pgo-instr-use-pipeline -profile-file='%t.profdata' \
; RUN:     -hot-cold-split \
; RUN:     -O2 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O2 --check-prefix=PGOUSE --check-prefix=SPLIT
;
; In the first pipeline there should just be a function pass manager, no other
; pass managers.
; CHECK-O2: Pass Arguments:
; CHECK-O2-NOT: Manager
; CHECK-O2: FunctionPass Manager
; CHECK-O2-NOT: Manager
;
; CHECK-O2: Pass Arguments:
; CHECK-O2: ModulePass Manager
; CHECK-O2-NOT: Manager
; First function pass pipeline just does early opts.
; CHECK-O2-COUNT-3: FunctionPass Manager
; CHECK-O2-NOT: Manager
; FIXME: It's a bit odd to do dead arg elim in the middle of early opts...
; CHECK-O2: Dead Argument Elimination
; CHECK-O2-NEXT: FunctionPass Manager
; CHECK-O2-NOT: Manager
; Very carefully assert the CGSCC pass pipeline as it is fragile and unusually
; susceptible to phase ordering issues.
; CHECK-O2: CallGraph Construction
; PGOUSE: Call Graph SCC Pass Manager
; PGOUSE:      Function Integration/Inlining
; PGOUSE: PGOInstrumentationUsePass
; PGOUSE: PGOIndirectCallPromotion
; PGOUSE: CallGraph Construction
; CHECK-O2-NEXT: Globals Alias Analysis
; CHECK-O2-NEXT: Call Graph SCC Pass Manager
; CHECK-O2-NEXT: Remove unused exception handling info
; CHECK-O2-NEXT: Function Integration/Inlining
; CHECK-O2-NEXT: OpenMP specific optimizations
; CHECK-O2-NEXT: Deduce function attributes
; Next up is the main function pass pipeline. It shouldn't be split up and
; should contain the main loop pass pipeline as well.
; CHECK-O2-NEXT: FunctionPass Manager
; CHECK-O2-NOT: Manager
; CHECK-O2: Loop Pass Manager
; CHECK-O2: Loop Pass Manager
; CHECK-O2-NOT: Manager
; FIXME: We shouldn't be pulling out to simplify-cfg and instcombine and
; causing new loop pass managers.
; CHECK-O2: Simplify the CFG
; CHECK-O2-NOT: Manager
; CHECK-O2: Combine redundant instructions
; CHECK-O2-NOT: Manager
; CHECK-O2: Loop Pass Manager
; CHECK-O2-NOT: Manager
; FIXME: It isn't clear that we need yet another loop pass pipeline
; and run of LICM here.
; CHECK-O2-NOT: Manager
; CHECK-O2: Loop Pass Manager
; CHECK-O2-NEXT: Loop Invariant Code Motion
; CHECK-O2-NOT: Manager
; Next we break out of the main Function passes inside the CGSCC pipeline with
; a barrier pass.
; CHECK-O2: A No-Op Barrier Pass
; CHECK-O2-NEXT: Eliminate Available Externally
; Inferring function attribute should be right after the CGSCC pipeline, before
; any other optimizations/analyses.
; CHECK-O2-NEXT: CallGraph
; CHECK-O2-NEXT: Deduce function attributes in RPO
; CHECK-O2-NOT: Manager
; Reduce the size of the IR ASAP after the inliner.
; CHECK-O2-NEXT: Global Variable Optimizer
; CHECK-O2: Dead Global Elimination
; Next is the late function pass pipeline.
; CHECK-O2: FunctionPass Manager
; CHECK-O2-NOT: Manager
; We rotate loops prior to vectorization.
; CHECK-O2: Loop Pass Manager
; CHECK-O2-NEXT: Rotate Loops
; CHECK-O2-NOT: Manager
; CHECK-O2: Loop Vectorization
; CHECK-O2-NOT: Manager
; CHECK-O2: SLP Vectorizer
; CHECK-O2-NOT: Manager
; After vectorization we do partial unrolling.
; CHECK-O2: Loop Pass Manager
; CHECK-O2-NEXT: Unroll loops
; CHECK-O2-NOT: Manager
; After vectorization and unrolling we try to do any cleanup of inserted code,
; including a run of LICM. This shouldn't run in the same loop pass manager as
; the runtime unrolling though.
; CHECK-O2: Loop Pass Manager
; CHECK-O2-NEXT: Loop Invariant Code Motion
; SPLIT: Hot Cold Splitting
; CHECK-O2: FunctionPass Manager
; CHECK-O2: Loop Pass Manager
; CHECK-O2-NEXT: Loop Sink
; CHECK-O2: Simplify the CFG
; CHECK-O2: Relative Lookup Table Converter
; CHECK-O2: FunctionPass Manager
; CHECK-O2-NOT: Manager
;
; FIXME: There really shouldn't be another pass manager, especially one that
; just builds the domtree. It doesn't even run the verifier.
; CHECK-O2: Pass Arguments:
; CHECK-O2: FunctionPass Manager
; CHECK-O2-NEXT: Dominator Tree Construction

define void @foo() {
  ret void
}
