; RUN: opt -enable-new-pm=0 -O3 -enable-matrix -debug-pass=Structure %s -disable-output 2>&1 | FileCheck --check-prefix=LEGACY %s
; RUN: opt -passes='default<O3>' -enable-matrix -debug-pass-manager -disable-output %s 2>&1 | FileCheck --check-prefix=NEWPM %s

; REQUIRES: asserts

; LEGACY-LABEL: Pass Arguments:
; LEGACY-NEXT: Target Transform Information
; LEGACY-NEXT: Type-Based Alias Analysis
; LEGACY-NEXT: Scoped NoAlias Alias Analysis
; LEGACY-NEXT: Assumption Cache Tracker
; LEGACY-NEXT: Target Library Information
; LEGACY-NEXT:   FunctionPass Manager
; LEGACY-NEXT:     Module Verifier
; LEGACY-EXT:      Good Bye World Pass
; LEGACY-NOEXT-NOT:      Good Bye World Pass
; LEGACY-NEXT:     Lower 'expect' Intrinsics
; LEGACY-NEXT:     Simplify the CFG
; LEGACY-NEXT:     Dominator Tree Construction
; LEGACY-NEXT:     SROA
; LEGACY-NEXT:     Early CSE
; LEGACY-NEXT: Pass Arguments:
; LEGACY-NEXT: Target Library Information
; LEGACY-NEXT: Target Transform Information
;             Target Pass Configuration
; LEGACY:      Type-Based Alias Analysis
; LEGACY-NEXT: Scoped NoAlias Alias Analysis
; LEGACY-NEXT: Assumption Cache Tracker
; LEGACY-NEXT: Profile summary info
; LEGACY-NEXT:   ModulePass Manager
; LEGACY-NEXT:     Annotation2Metadata
; LEGACY-NEXT:     Force set function attributes
; LEGACY-NEXT:     Infer set function attributes
; LEGACY-NEXT:     FunctionPass Manager
; LEGACY-NEXT:       Dominator Tree Construction
; LEGACY-NEXT:       Call-site splitting
; LEGACY-NEXT:     Interprocedural Sparse Conditional Constant Propagation
; LEGACY-NEXT:       FunctionPass Manager
; LEGACY-NEXT:         Dominator Tree Construction
; LEGACY-NEXT:     Called Value Propagation
; LEGACY-NEXT:     Global Variable Optimizer
; LEGACY-NEXT:       FunctionPass Manager
; LEGACY-NEXT:         Dominator Tree Construction
; LEGACY-NEXT:         Natural Loop Information
; LEGACY-NEXT:         Post-Dominator Tree Construction
; LEGACY-NEXT:         Branch Probability Analysis
; LEGACY-NEXT:         Block Frequency Analysis
; LEGACY-NEXT:     FunctionPass Manager
; LEGACY-NEXT:       Dominator Tree Construction
; LEGACY-NEXT:       Promote Memory to Register
; LEGACY-NEXT:     Dead Argument Elimination
; LEGACY-NEXT:     FunctionPass Manager
; LEGACY-NEXT:       Dominator Tree Construction
; LEGACY-NEXT:       Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:       Function Alias Analysis Results
; LEGACY-NEXT:       Natural Loop Information
; LEGACY-NEXT:       Lazy Branch Probability Analysis
; LEGACY-NEXT:       Lazy Block Frequency Analysis
; LEGACY-NEXT:       Optimization Remark Emitter
; LEGACY-NEXT:       Combine redundant instructions
; LEGACY-NEXT:       Simplify the CFG
; LEGACY-NEXT:     CallGraph Construction
; LEGACY-NEXT:     Globals Alias Analysis
; LEGACY-NEXT:     Call Graph SCC Pass Manager
; LEGACY-NEXT:       Remove unused exception handling info
; LEGACY-NEXT:       Function Integration/Inlining
; LEGACY-NEXT:       OpenMP specific optimizations
; LEGACY-NEXT:       Deduce function attributes
; LEGACY-NEXT:       Promote 'by reference' arguments to scalars
; LEGACY-NEXT:       FunctionPass Manager
; LEGACY-NEXT:         Dominator Tree Construction
; LEGACY-NEXT:         SROA
; LEGACY-NEXT:         Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:         Function Alias Analysis Results
; LEGACY-NEXT:         Memory SSA
; LEGACY-NEXT:         Early CSE w/ MemorySSA
; LEGACY-NEXT:         Speculatively execute instructions if target has divergent branches
; LEGACY-NEXT:         Function Alias Analysis Results
; LEGACY-NEXT:         Lazy Value Information Analysis
; LEGACY-NEXT:         Jump Threading
; LEGACY-NEXT:         Value Propagation
; LEGACY-NEXT:         Simplify the CFG
; LEGACY-NEXT:         Dominator Tree Construction
; LEGACY-NEXT:         Combine pattern based expressions
; LEGACY-NEXT:         Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:         Function Alias Analysis Results
; LEGACY-NEXT:         Natural Loop Information
; LEGACY-NEXT:         Lazy Branch Probability Analysis
; LEGACY-NEXT:         Lazy Block Frequency Analysis
; LEGACY-NEXT:         Optimization Remark Emitter
; LEGACY-NEXT:         Combine redundant instructions
; LEGACY-NEXT:         Conditionally eliminate dead library calls
; LEGACY-NEXT:         Natural Loop Information
; LEGACY-NEXT:         Post-Dominator Tree Construction
; LEGACY-NEXT:         Branch Probability Analysis
; LEGACY-NEXT:         Block Frequency Analysis
; LEGACY-NEXT:         Lazy Branch Probability Analysis
; LEGACY-NEXT:         Lazy Block Frequency Analysis
; LEGACY-NEXT:         Optimization Remark Emitter
; LEGACY-NEXT:         PGOMemOPSize
; LEGACY-NEXT:         Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:         Function Alias Analysis Results
; LEGACY-NEXT:         Natural Loop Information
; LEGACY-NEXT:         Lazy Branch Probability Analysis
; LEGACY-NEXT:         Lazy Block Frequency Analysis
; LEGACY-NEXT:         Optimization Remark Emitter
; LEGACY-NEXT:         Tail Call Elimination
; LEGACY-NEXT:         Simplify the CFG
; LEGACY-NEXT:         Reassociate expressions
; LEGACY-NEXT:         Dominator Tree Construction
; LEGACY-NEXT:         Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:         Function Alias Analysis Results
; LEGACY-NEXT:         Memory SSA
; LEGACY-NEXT:         Natural Loop Information
; LEGACY-NEXT:         Canonicalize natural loops
; LEGACY-NEXT:         LCSSA Verifier
; LEGACY-NEXT:         Loop-Closed SSA Form Pass
; LEGACY-NEXT:         Scalar Evolution Analysis
; LEGACY-NEXT:         Lazy Branch Probability Analysis
; LEGACY-NEXT:         Lazy Block Frequency Analysis
; LEGACY-NEXT:         Loop Pass Manager
; LEGACY-NEXT:           Loop Invariant Code Motion
; LEGACY-NEXT:           Rotate Loops
; LEGACY-NEXT:           Loop Invariant Code Motion
; LEGACY-NEXT:           Unswitch loops
; LEGACY-NEXT:         Simplify the CFG
; LEGACY-NEXT:         Dominator Tree Construction
; LEGACY-NEXT:         Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:         Function Alias Analysis Results
; LEGACY-NEXT:         Natural Loop Information
; LEGACY-NEXT:         Lazy Branch Probability Analysis
; LEGACY-NEXT:         Lazy Block Frequency Analysis
; LEGACY-NEXT:         Optimization Remark Emitter
; LEGACY-NEXT:         Combine redundant instructions
; LEGACY-NEXT:         Canonicalize natural loops
; LEGACY-NEXT:         LCSSA Verifier
; LEGACY-NEXT:         Loop-Closed SSA Form Pass
; LEGACY-NEXT:         Scalar Evolution Analysis
; LEGACY-NEXT:         Loop Pass Manager
; LEGACY-NEXT:           Recognize loop idioms
; LEGACY-NEXT:           Induction Variable Simplification
; LEGACY-NEXT:           Delete dead loops
; LEGACY-NEXT:           Unroll loops
; LEGACY-NEXT:         SROA
; LEGACY-NEXT:         Function Alias Analysis Results
; LEGACY-NEXT:         MergedLoadStoreMotion
; LEGACY-NEXT:         Phi Values Analysis
; LEGACY-NEXT:         Function Alias Analysis Results
; LEGACY-NEXT:         Memory Dependence Analysis
; LEGACY-NEXT:         Lazy Branch Probability Analysis
; LEGACY-NEXT:         Lazy Block Frequency Analysis
; LEGACY-NEXT:         Optimization Remark Emitter
; LEGACY-NEXT:         Global Value Numbering
; LEGACY-NEXT:         Sparse Conditional Constant Propagation
; LEGACY-NEXT:         Demanded bits analysis
; LEGACY-NEXT:         Bit-Tracking Dead Code Elimination
; LEGACY-NEXT:         Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:         Function Alias Analysis Results
; LEGACY-NEXT:         Lazy Branch Probability Analysis
; LEGACY-NEXT:         Lazy Block Frequency Analysis
; LEGACY-NEXT:         Optimization Remark Emitter
; LEGACY-NEXT:         Combine redundant instructions
; LEGACY-NEXT:         Lazy Value Information Analysis
; LEGACY-NEXT:         Jump Threading
; LEGACY-NEXT:         Value Propagation
; LEGACY-NEXT:         Post-Dominator Tree Construction
; LEGACY-NEXT:         Aggressive Dead Code Elimination
; LEGACY-NEXT:         Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:         Function Alias Analysis Results
; LEGACY-NEXT:         Memory SSA
; LEGACY-NEXT:         MemCpy Optimization
; LEGACY-NEXT:         Dead Store Elimination
; LEGACY-NEXT:         Natural Loop Information
; LEGACY-NEXT:         Canonicalize natural loops
; LEGACY-NEXT:         LCSSA Verifier
; LEGACY-NEXT:         Loop-Closed SSA Form Pass
; LEGACY-NEXT:         Function Alias Analysis Results
; LEGACY-NEXT:         Scalar Evolution Analysis
; LEGACY-NEXT:         Lazy Branch Probability Analysis
; LEGACY-NEXT:         Lazy Block Frequency Analysis
; LEGACY-NEXT:         Loop Pass Manager
; LEGACY-NEXT:           Loop Invariant Code Motion
; LEGACY-NEXT:         Simplify the CFG
; LEGACY-NEXT:         Dominator Tree Construction
; LEGACY-NEXT:         Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:         Function Alias Analysis Results
; LEGACY-NEXT:         Natural Loop Information
; LEGACY-NEXT:         Lazy Branch Probability Analysis
; LEGACY-NEXT:         Lazy Block Frequency Analysis
; LEGACY-NEXT:         Optimization Remark Emitter
; LEGACY-NEXT:         Combine redundant instructions
; LEGACY-NEXT:     A No-Op Barrier Pass
; LEGACY-NEXT:     Eliminate Available Externally Globals
; LEGACY-NEXT:     CallGraph Construction
; LEGACY-NEXT:     Deduce function attributes in RPO
; LEGACY-NEXT:     Global Variable Optimizer
; LEGACY-NEXT:       FunctionPass Manager
; LEGACY-NEXT:         Dominator Tree Construction
; LEGACY-NEXT:         Natural Loop Information
; LEGACY-NEXT:         Post-Dominator Tree Construction
; LEGACY-NEXT:         Branch Probability Analysis
; LEGACY-NEXT:         Block Frequency Analysis
; LEGACY-NEXT:     Dead Global Elimination
; LEGACY-NEXT:     CallGraph Construction
; LEGACY-NEXT:     Globals Alias Analysis
; LEGACY-NEXT:     FunctionPass Manager
; LEGACY-NEXT:       Dominator Tree Construction
; LEGACY-NEXT:       Float to int
; LEGACY-NEXT:       Lower constant intrinsics
; LEGACY-NEXT:       Natural Loop Information
; LEGACY-NEXT:       Lazy Branch Probability Analysis
; LEGACY-NEXT:       Lazy Block Frequency Analysis
; LEGACY-NEXT:       Optimization Remark Emitter
; LEGACY-NEXT:       Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:       Function Alias Analysis Results
; LEGACY-NEXT:       Lower the matrix intrinsics
; LEGACY-NEXT:       Early CSE
; LEGACY-NEXT:       Canonicalize natural loops
; LEGACY-NEXT:       LCSSA Verifier
; LEGACY-NEXT:       Loop-Closed SSA Form Pass
; LEGACY-NEXT:       Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:       Function Alias Analysis Results
; LEGACY-NEXT:       Scalar Evolution Analysis
; LEGACY-NEXT:       Loop Pass Manager
; LEGACY-NEXT:         Rotate Loops
; LEGACY-NEXT:       Loop Access Analysis
; LEGACY-NEXT:       Lazy Branch Probability Analysis
; LEGACY-NEXT:       Lazy Block Frequency Analysis
; LEGACY-NEXT:       Optimization Remark Emitter
; LEGACY-NEXT:       Loop Distribution
; LEGACY-NEXT:       Post-Dominator Tree Construction
; LEGACY-NEXT:       Branch Probability Analysis
; LEGACY-NEXT:       Block Frequency Analysis
; LEGACY-NEXT:       Scalar Evolution Analysis
; LEGACY-NEXT:       Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:       Function Alias Analysis Results
; LEGACY-NEXT:       Loop Access Analysis
; LEGACY-NEXT:       Demanded bits analysis
; LEGACY-NEXT:       Lazy Branch Probability Analysis
; LEGACY-NEXT:       Lazy Block Frequency Analysis
; LEGACY-NEXT:       Optimization Remark Emitter
; LEGACY-NEXT:       Inject TLI Mappings
; LEGACY-NEXT:       Loop Vectorization
; LEGACY-NEXT:       Canonicalize natural loops
; LEGACY-NEXT:       Scalar Evolution Analysis
; LEGACY-NEXT:       Function Alias Analysis Results
; LEGACY-NEXT:       Loop Access Analysis
; LEGACY-NEXT:       Lazy Branch Probability Analysis
; LEGACY-NEXT:       Lazy Block Frequency Analysis
; LEGACY-NEXT:       Loop Load Elimination
; LEGACY-NEXT:       Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:       Function Alias Analysis Results
; LEGACY-NEXT:       Lazy Branch Probability Analysis
; LEGACY-NEXT:       Lazy Block Frequency Analysis
; LEGACY-NEXT:       Optimization Remark Emitter
; LEGACY-NEXT:       Combine redundant instructions
; LEGACY-NEXT:       Simplify the CFG
; LEGACY-NEXT:       Dominator Tree Construction
; LEGACY-NEXT:       Natural Loop Information
; LEGACY-NEXT:       Scalar Evolution Analysis
; LEGACY-NEXT:       Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:       Function Alias Analysis Results
; LEGACY-NEXT:       Demanded bits analysis
; LEGACY-NEXT:       Lazy Branch Probability Analysis
; LEGACY-NEXT:       Lazy Block Frequency Analysis
; LEGACY-NEXT:       Optimization Remark Emitter
; LEGACY-NEXT:       Inject TLI Mappings
; LEGACY-NEXT:       SLP Vectorizer
; LEGACY-NEXT:       Optimize scalar/vector ops
; LEGACY-NEXT:       Optimization Remark Emitter
; LEGACY-NEXT:       Combine redundant instructions
; LEGACY-NEXT:       Canonicalize natural loops
; LEGACY-NEXT:       LCSSA Verifier
; LEGACY-NEXT:       Loop-Closed SSA Form Pass
; LEGACY-NEXT:       Scalar Evolution Analysis
; LEGACY-NEXT:       Loop Pass Manager
; LEGACY-NEXT:         Unroll loops
; LEGACY-NEXT:       Lazy Branch Probability Analysis
; LEGACY-NEXT:       Lazy Block Frequency Analysis
; LEGACY-NEXT:       Optimization Remark Emitter
; LEGACY-NEXT:       Combine redundant instructions
; LEGACY-NEXT:       Memory SSA
; LEGACY-NEXT:       Canonicalize natural loops
; LEGACY-NEXT:       LCSSA Verifier
; LEGACY-NEXT:       Loop-Closed SSA Form Pass
; LEGACY-NEXT:       Scalar Evolution Analysis
; LEGACY-NEXT:       Lazy Branch Probability Analysis
; LEGACY-NEXT:       Lazy Block Frequency Analysis
; LEGACY-NEXT:       Loop Pass Manager
; LEGACY-NEXT:         Loop Invariant Code Motion
; LEGACY-NEXT:       Optimization Remark Emitter
; LEGACY-NEXT:       Warn about non-applied transformations
; LEGACY-NEXT:       Alignment from assumptions
; LEGACY-NEXT:     Strip Unused Function Prototypes
; LEGACY-NEXT:     Dead Global Elimination
; LEGACY-NEXT:     Merge Duplicate Global Constants
; LEGACY-NEXT:     Call Graph Profile
; LEGACY-NEXT:       FunctionPass Manager
; LEGACY-NEXT:         Dominator Tree Construction
; LEGACY-NEXT:         Natural Loop Information
; LEGACY-NEXT:         Lazy Branch Probability Analysis
; LEGACY-NEXT:         Lazy Block Frequency Analysis
; LEGACY-NEXT:     FunctionPass Manager
; LEGACY-NEXT:       Dominator Tree Construction
; LEGACY-NEXT:       Natural Loop Information
; LEGACY-NEXT:       Post-Dominator Tree Construction
; LEGACY-NEXT:       Branch Probability Analysis
; LEGACY-NEXT:       Block Frequency Analysis
; LEGACY-NEXT:       Canonicalize natural loops
; LEGACY-NEXT:       LCSSA Verifier
; LEGACY-NEXT:       Loop-Closed SSA Form Pass
; LEGACY-NEXT:       Basic Alias Analysis (stateless AA impl)
; LEGACY-NEXT:       Function Alias Analysis Results
; LEGACY-NEXT:       Scalar Evolution Analysis
; LEGACY-NEXT:       Block Frequency Analysis
; LEGACY-NEXT:       Loop Pass Manager
; LEGACY-NEXT:         Loop Sink
; LEGACY-NEXT:       Lazy Branch Probability Analysis
; LEGACY-NEXT:       Lazy Block Frequency Analysis
; LEGACY-NEXT:       Optimization Remark Emitter
; LEGACY-NEXT:       Remove redundant instructions
; LEGACY-NEXT:       Hoist/decompose integer division and remainder
; LEGACY-NEXT:       Simplify the CFG
; LEGACY-NEXT:       Annotation Remarks
; LEGACY-NEXT:       Module Verifier
; LEGACY-NEXT: Pass Arguments:
; LEGACY-NEXT:  FunctionPass Manager
; LEGACY-NEXT:     Dominator Tree Construction
; LEGACY-NEXT: Pass Arguments:
; LEGACY-NEXT: Target Library Information
; LEGACY-NEXT:   FunctionPass Manager
; LEGACY-NEXT:     Dominator Tree Construction
; LEGACY-NEXT:     Natural Loop Information
; LEGACY-NEXT:     Post-Dominator Tree Construction
; LEGACY-NEXT:     Branch Probability Analysis
; LEGACY-NEXT:     Block Frequency Analysis
; LEGACY-NEXT: Pass Arguments:
; LEGACY-NEXT: Target Library Information
; LEGACY-NEXT:   FunctionPass Manager
; LEGACY-NEXT:     Dominator Tree Construction
; LEGACY-NEXT:     Natural Loop Information
; LEGACY-NEXT:     Post-Dominator Tree Construction
; LEGACY-NEXT:     Branch Probability Analysis
; LEGACY-NEXT:     Block Frequency Analysis

; NEWPM:      Running pass: VerifierPass on
; NEWPM-NEXT: Running analysis: VerifierAnalysis on
; NEWPM-NEXT:Running pass: Annotation2MetadataPass on
; NEWPM-NEXT:Running pass: ForceFunctionAttrsPass on
; NEWPM-NEXT:Running pass: InferFunctionAttrsPass on
; NEWPM-NEXT:Running analysis: InnerAnalysisManagerProxy<
; NEWPM-NEXT:Running analysis: PreservedCFGCheckerAnalysis on f
; NEWPM-NEXT:Running pass: LowerExpectIntrinsicPass on f
; NEWPM-NEXT: Running pass: SimplifyCFGPass on f
; NEWPM-NEXT: Running analysis: TargetIRAnalysis on f
; NEWPM-NEXT: Running analysis: AssumptionAnalysis on f
; NEWPM-NEXT: Running pass: SROA on f
; NEWPM-NEXT: Running analysis: DominatorTreeAnalysis on f
; NEWPM-NEXT: Running pass: EarlyCSEPass on f
; NEWPM-NEXT: Running analysis: TargetLibraryAnalysis on f
; NEWPM-NEXT: Running pass: CallSiteSplittingPass on f
; NEWPM-NEXT: Running pass: OpenMPOptPass on
; NEWPM-NEXT: Running pass: IPSCCPPass on
; NEWPM-NEXT: Running pass: CalledValuePropagationPass on
; NEWPM-NEXT: Running pass: GlobalOptPass on
; NEWPM-NEXT: Invalidating analysis: VerifierAnalysis on
; NEWPM-NEXT: Invalidating analysis: InnerAnalysisManagerProxy<
; NEWPM-NEXT: Running analysis: InnerAnalysisManagerProxy<
; NEWPM-NEXT: Running pass: PromotePass on f
; NEWPM-NEXT: Running analysis: PreservedCFGCheckerAnalysis on f
; NEWPM-NEXT: Running analysis: DominatorTreeAnalysis on f
; NEWPM-NEXT: Running analysis: AssumptionAnalysis on f
; NEWPM-NEXT: Running pass: DeadArgumentEliminationPass on
; NEWPM-NEXT: Running pass: InstCombinePass on f
; NEWPM-NEXT: Running analysis: TargetLibraryAnalysis on f
; NEWPM-NEXT: Running analysis: OptimizationRemarkEmitterAnalysis on f
; NEWPM-NEXT: Running analysis: TargetIRAnalysis on f
; NEWPM-NEXT: Running analysis: AAManager on f
; NEWPM-NEXT: Running analysis: BasicAA on f
; NEWPM-NEXT: Running analysis: ScopedNoAliasAA on f
; NEWPM-NEXT: Running analysis: TypeBasedAA on f
; NEWPM-NEXT: Running analysis: OuterAnalysisManagerProxy<
; NEWPM-NEXT: Running pass: SimplifyCFGPass on f
; NEWPM-NEXT: Running pass: ModuleInlinerWrapperPass on
; NEWPM-NEXT: Running analysis: InlineAdvisorAnalysis on
; NEWPM-NEXT: Running pass: RequireAnalysisPass<{{.*}}GlobalsAA
; NEWPM-NEXT: Running analysis: GlobalsAA on
; NEWPM-NEXT: Running analysis: CallGraphAnalysis on
; NEWPM-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}AAManager
; NEWPM-NEXT: Invalidating analysis: AAManager on f
; NEWPM-NEXT: Running pass: RequireAnalysisPass<{{.*}}ProfileSummaryAnalysis
; NEWPM-NEXT: Running analysis: ProfileSummaryAnalysis on
; NEWPM-NEXT: Running analysis: InnerAnalysisManagerProxy<
; NEWPM-NEXT: Running analysis: LazyCallGraphAnalysis on
; NEWPM-NEXT: Running analysis: FunctionAnalysisManagerCGSCCProxy on (f)
; NEWPM-NEXT: Running analysis: OuterAnalysisManagerProxy<
; NEWPM-NEXT: Running pass: DevirtSCCRepeatedPass on (f)
; NEWPM-NEXT: Running pass: InlinerPass on (f)
; NEWPM-NEXT: Running pass: InlinerPass on (f)
; NEWPM-NEXT: Running pass: PostOrderFunctionAttrsPass on (f)
; NEWPM-NEXT: Running analysis: AAManager on f
; NEWPM-NEXT: Running pass: ArgumentPromotionPass on (f)
; NEWPM-NEXT: Running pass: OpenMPOptCGSCCPass on (f)
; NEWPM-NEXT: Running pass: SROA on f
; NEWPM-NEXT: Running pass: EarlyCSEPass on f
; NEWPM-NEXT: Running analysis: MemorySSAAnalysis on f
; NEWPM-NEXT: Running pass: SpeculativeExecutionPass on f
; NEWPM-NEXT: Running pass: JumpThreadingPass on f
; NEWPM-NEXT: Running analysis: LazyValueAnalysis on f
; NEWPM-NEXT: Running pass: CorrelatedValuePropagationPass on f
; NEWPM-NEXT: Invalidating analysis: LazyValueAnalysis on f
; NEWPM-NEXT: Running pass: SimplifyCFGPass on f
; NEWPM-NEXT: Running pass: AggressiveInstCombinePass on f
; NEWPM-NEXT: Running pass: InstCombinePass on f
; NEWPM-NEXT: Running pass: LibCallsShrinkWrapPass on f
; NEWPM-NEXT: Running pass: TailCallElimPass on f
; NEWPM-NEXT: Running pass: SimplifyCFGPass on f
; NEWPM-NEXT: Running pass: ReassociatePass on f
; NEWPM-NEXT: Running pass: RequireAnalysisPass<{{.*}}OptimizationRemarkEmitterAnalysis
; NEWPM-NEXT: Running pass: LoopSimplifyPass on f
; NEWPM-NEXT: Running analysis: LoopAnalysis on f
; NEWPM-NEXT: Running pass: LCSSAPass on f
; NEWPM-NEXT: Running pass: SimplifyCFGPass on f
; NEWPM-NEXT: Running pass: InstCombinePass on f
; NEWPM-NEXT: Running pass: LoopSimplifyPass on f
; NEWPM-NEXT: Running pass: LCSSAPass on f
; NEWPM-NEXT: Running pass: SROA on f
; NEWPM-NEXT: Running pass: MergedLoadStoreMotionPass on f
; NEWPM-NEXT: Running pass: GVN on f
; NEWPM-NEXT: Running analysis: MemoryDependenceAnalysis on f
; NEWPM-NEXT: Running analysis: PhiValuesAnalysis on f
; NEWPM-NEXT: Running pass: SCCPPass on f
; NEWPM-NEXT: Running pass: BDCEPass on f
; NEWPM-NEXT: Running analysis: DemandedBitsAnalysis on f
; NEWPM-NEXT: Running pass: InstCombinePass on f
; NEWPM-NEXT: Running pass: JumpThreadingPass on f
; NEWPM-NEXT: Running analysis: LazyValueAnalysis on f
; NEWPM-NEXT: Running pass: CorrelatedValuePropagationPass on f
; NEWPM-NEXT: Invalidating analysis: LazyValueAnalysis on f
; NEWPM-NEXT: Running pass: ADCEPass on f
; NEWPM-NEXT: Running analysis: PostDominatorTreeAnalysis on f
; NEWPM-NEXT: Running pass: MemCpyOptPass on f
; NEWPM-NEXT: Running pass: DSEPass on f
; NEWPM-NEXT: Running pass: LoopSimplifyPass on f
; NEWPM-NEXT: Running pass: LCSSAPass on f
; NEWPM-NEXT: Running pass: SimplifyCFGPass on f
; NEWPM-NEXT: Running pass: InstCombinePass on f
; NEWPM-NEXT: Invalidating analysis: CallGraphAnalysis on
; NEWPM-NEXT: Running pass: GlobalOptPass on
; NEWPM-NEXT: Running pass: GlobalDCEPass on
; NEWPM-NEXT: Running pass: EliminateAvailableExternallyPass on
; NEWPM-NEXT: Running pass: ReversePostOrderFunctionAttrsPass on
; NEWPM-NEXT: Running analysis: CallGraphAnalysis on
; NEWPM-NEXT: Running pass: RequireAnalysisPass<{{.*}}GlobalsAA
; NEWPM-NEXT: Running pass: Float2IntPass on f
; NEWPM-NEXT: Running pass: LowerConstantIntrinsicsPass on f
; NEWPM-NEXT: Running pass: LowerMatrixIntrinsicsPass on f
; NEWPM-NEXT: Running pass: EarlyCSEPass on f
; NEWPM-NEXT: Running pass: LoopSimplifyPass on f
; NEWPM-NEXT: Running pass: LCSSAPass on f
; NEWPM-NEXT: Running pass: LoopDistributePass on f
; NEWPM-NEXT: Running analysis: ScalarEvolutionAnalysis on f
; NEWPM-NEXT: Running analysis: InnerAnalysisManagerProxy<
; NEWPM-NEXT: Running pass: InjectTLIMappings on f
; NEWPM-NEXT: Running pass: LoopVectorizePass on f
; NEWPM-NEXT: Running analysis: BlockFrequencyAnalysis on f
; NEWPM-NEXT: Running analysis: BranchProbabilityAnalysis on f
; NEWPM-NEXT: Running pass: LoopLoadEliminationPass on f
; NEWPM-NEXT: Running pass: InstCombinePass on f
; NEWPM-NEXT: Running pass: SimplifyCFGPass on f
; NEWPM-NEXT: Running pass: SLPVectorizerPass on f
; NEWPM-NEXT: Running pass: VectorCombinePass on f
; NEWPM-NEXT: Running pass: InstCombinePass on f
; NEWPM-NEXT: Running pass: LoopUnrollPass on f
; NEWPM-NEXT: Running pass: WarnMissedTransformationsPass on f
; NEWPM-NEXT: Running pass: InstCombinePass on f
; NEWPM-NEXT: Running pass: RequireAnalysisPass<{{.*}}OptimizationRemarkEmitterAnalysis
; NEWPM-NEXT: Running pass: LoopSimplifyPass on f
; NEWPM-NEXT: Running pass: LCSSAPass on f
; NEWPM-NEXT: Running pass: AlignmentFromAssumptionsPass on f
; NEWPM-NEXT: Running pass: LoopSinkPass on f
; NEWPM-NEXT: Running pass: InstSimplifyPass on f
; NEWPM-NEXT: Running pass: DivRemPairsPass on f
; NEWPM-NEXT: Running pass: SimplifyCFGPass on f
; NEWPM-NEXT: Running pass: SpeculateAroundPHIsPass on f
; NEWPM-NEXT: Running pass: CGProfilePass on
; NEWPM-NEXT: Running pass: GlobalDCEPass on
; NEWPM-NEXT: Running pass: ConstantMergePass on
; NEWPM-NEXT: Running pass: RelLookupTableConverterPass on
; NEWPM-NEXT: Running pass: AnnotationRemarksPass on f
; NEWPM-NEXT: Running pass: VerifierPass on
; NEWPM-NEXT: Running analysis: VerifierAnalysis on

define void @f() {
  ret void
}
