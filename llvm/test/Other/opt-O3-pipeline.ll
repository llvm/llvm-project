; RUN: opt -enable-new-pm=0 -mtriple=x86_64-- -O3 -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck --check-prefixes=CHECK,%llvmcheckext %s
; RUN: opt -enable-new-pm=1 -mtriple=x86_64-- -O3 -debug-pass-structure < %s -o /dev/null 2>&1 | FileCheck --check-prefixes=NEWPM,%llvmcheckext %s

; REQUIRES: asserts

; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Transform Information
; CHECK-NEXT: Type-Based Alias Analysis
; CHECK-NEXT: Scoped NoAlias Alias Analysis
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-NEXT: Target Library Information
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Module Verifier
; CHECK-EXT:      Good Bye World Pass
; CHECK-NOEXT-NOT:      Good Bye World Pass
; CHECK-NEXT:     Lower 'expect' Intrinsics
; CHECK-NEXT:     Simplify the CFG
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT:     SROA
; CHECK-NEXT:     Early CSE
; CHECK-NEXT: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: Target Transform Information
;             Target Pass Configuration
; CHECK:      Type-Based Alias Analysis
; CHECK-NEXT: Scoped NoAlias Alias Analysis
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-NEXT: Profile summary info
; CHECK-NEXT:   ModulePass Manager
; CHECK-NEXT:     Annotation2Metadata
; CHECK-NEXT:     Force set function attributes
; CHECK-NEXT:     Infer set function attributes
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Call-site splitting
; CHECK-NEXT:     Interprocedural Sparse Conditional Constant Propagation
; CHECK-NEXT:       FunctionPass Manager
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:     Called Value Propagation
; CHECK-NEXT:     Global Variable Optimizer
; CHECK-NEXT:       FunctionPass Manager
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Post-Dominator Tree Construction
; CHECK-NEXT:         Branch Probability Analysis
; CHECK-NEXT:         Block Frequency Analysis
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Promote Memory to Register
; CHECK-NEXT:     Dead Argument Elimination
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Combine redundant instructions
; CHECK-NEXT:       Simplify the CFG
; CHECK-NEXT:     CallGraph Construction
; CHECK-NEXT:     Globals Alias Analysis
; CHECK-NEXT:     Call Graph SCC Pass Manager
; CHECK-NEXT:       Remove unused exception handling info
; CHECK-NEXT:       Function Integration/Inlining
; CHECK-NEXT:       OpenMP specific optimizations
; CHECK-NEXT:       Deduce function attributes
; CHECK-NEXT:       Promote 'by reference' arguments to scalars
; CHECK-NEXT:       FunctionPass Manager
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:         SROA
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Memory SSA
; CHECK-NEXT:         Early CSE w/ MemorySSA
; CHECK-NEXT:         Speculatively execute instructions if target has divergent branches
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Lazy Value Information Analysis
; CHECK-NEXT:         Jump Threading
; CHECK-NEXT:         Value Propagation
; CHECK-NEXT:         Simplify the CFG
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:         Combine pattern based expressions
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:         Optimization Remark Emitter
; CHECK-NEXT:         Combine redundant instructions
; CHECK-NEXT:         Conditionally eliminate dead library calls
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Post-Dominator Tree Construction
; CHECK-NEXT:         Branch Probability Analysis
; CHECK-NEXT:         Block Frequency Analysis
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:         Optimization Remark Emitter
; CHECK-NEXT:         PGOMemOPSize
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:         Optimization Remark Emitter
; CHECK-NEXT:         Tail Call Elimination
; CHECK-NEXT:         Simplify the CFG
; CHECK-NEXT:         Reassociate expressions
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Memory SSA
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Canonicalize natural loops
; CHECK-NEXT:         LCSSA Verifier
; CHECK-NEXT:         Loop-Closed SSA Form Pass
; CHECK-NEXT:         Scalar Evolution Analysis
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:         Loop Pass Manager
; CHECK-NEXT:           Loop Invariant Code Motion
; CHECK-NEXT:           Rotate Loops
; CHECK-NEXT:           Loop Invariant Code Motion
; CHECK-NEXT:           Unswitch loops
; CHECK-NEXT:         Simplify the CFG
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:         Optimization Remark Emitter
; CHECK-NEXT:         Combine redundant instructions
; CHECK-NEXT:         Canonicalize natural loops
; CHECK-NEXT:         LCSSA Verifier
; CHECK-NEXT:         Loop-Closed SSA Form Pass
; CHECK-NEXT:         Scalar Evolution Analysis
; CHECK-NEXT:         Loop Pass Manager
; CHECK-NEXT:           Recognize loop idioms
; CHECK-NEXT:           Induction Variable Simplification
; CHECK-NEXT:           Delete dead loops
; CHECK-NEXT:           Unroll loops
; CHECK-NEXT:         SROA
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         MergedLoadStoreMotion
; CHECK-NEXT:         Phi Values Analysis
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Memory Dependence Analysis
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:         Optimization Remark Emitter
; CHECK-NEXT:         Global Value Numbering
; CHECK-NEXT:         Sparse Conditional Constant Propagation
; CHECK-NEXT:         Demanded bits analysis
; CHECK-NEXT:         Bit-Tracking Dead Code Elimination
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:         Optimization Remark Emitter
; CHECK-NEXT:         Combine redundant instructions
; CHECK-NEXT:         Lazy Value Information Analysis
; CHECK-NEXT:         Jump Threading
; CHECK-NEXT:         Value Propagation
; CHECK-NEXT:         Post-Dominator Tree Construction
; CHECK-NEXT:         Aggressive Dead Code Elimination
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Memory SSA
; CHECK-NEXT:         MemCpy Optimization
; CHECK-NEXT:         Dead Store Elimination
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Canonicalize natural loops
; CHECK-NEXT:         LCSSA Verifier
; CHECK-NEXT:         Loop-Closed SSA Form Pass
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Scalar Evolution Analysis
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:         Loop Pass Manager
; CHECK-NEXT:           Loop Invariant Code Motion
; CHECK-NEXT:         Simplify the CFG
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:         Optimization Remark Emitter
; CHECK-NEXT:         Combine redundant instructions
; CHECK-NEXT:     A No-Op Barrier Pass
; CHECK-NEXT:     Eliminate Available Externally Globals
; CHECK-NEXT:     CallGraph Construction
; CHECK-NEXT:     Deduce function attributes in RPO
; CHECK-NEXT:     Global Variable Optimizer
; CHECK-NEXT:       FunctionPass Manager
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Post-Dominator Tree Construction
; CHECK-NEXT:         Branch Probability Analysis
; CHECK-NEXT:         Block Frequency Analysis
; CHECK-NEXT:     Dead Global Elimination
; CHECK-NEXT:     CallGraph Construction
; CHECK-NEXT:     Globals Alias Analysis
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Float to int
; CHECK-NEXT:       Lower constant intrinsics
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       LCSSA Verifier
; CHECK-NEXT:       Loop-Closed SSA Form Pass
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Loop Pass Manager
; CHECK-NEXT:         Rotate Loops
; CHECK-NEXT:       Loop Access Analysis
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Loop Distribution
; CHECK-NEXT:       Post-Dominator Tree Construction
; CHECK-NEXT:       Branch Probability Analysis
; CHECK-NEXT:       Block Frequency Analysis
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Loop Access Analysis
; CHECK-NEXT:       Demanded bits analysis
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Inject TLI Mappings
; CHECK-NEXT:       Loop Vectorization
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Loop Access Analysis
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Loop Load Elimination
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Combine redundant instructions
; CHECK-NEXT:       Simplify the CFG
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Demanded bits analysis
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Inject TLI Mappings
; CHECK-NEXT:       SLP Vectorizer
; CHECK-NEXT:       Optimize scalar/vector ops
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Combine redundant instructions
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       LCSSA Verifier
; CHECK-NEXT:       Loop-Closed SSA Form Pass
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Loop Pass Manager
; CHECK-NEXT:         Unroll loops
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Combine redundant instructions
; CHECK-NEXT:       Memory SSA
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       LCSSA Verifier
; CHECK-NEXT:       Loop-Closed SSA Form Pass
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Loop Pass Manager
; CHECK-NEXT:         Loop Invariant Code Motion
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Warn about non-applied transformations
; CHECK-NEXT:       Alignment from assumptions
; CHECK-NEXT:     Strip Unused Function Prototypes
; CHECK-NEXT:     Dead Global Elimination
; CHECK-NEXT:     Merge Duplicate Global Constants
; CHECK-NEXT:     Call Graph Profile
; CHECK-NEXT:       FunctionPass Manager
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Post-Dominator Tree Construction
; CHECK-NEXT:       Branch Probability Analysis
; CHECK-NEXT:       Block Frequency Analysis
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       LCSSA Verifier
; CHECK-NEXT:       Loop-Closed SSA Form Pass
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Block Frequency Analysis
; CHECK-NEXT:       Loop Pass Manager
; CHECK-NEXT:         Loop Sink
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Remove redundant instructions
; CHECK-NEXT:       Hoist/decompose integer division and remainder
; CHECK-NEXT:       Simplify the CFG
; CHECK-NEXT:       Annotation Remarks
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:     Bitcode Writer
; CHECK-NEXT: Pass Arguments:
; CHECK-NEXT:  FunctionPass Manager
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT:     Natural Loop Information
; CHECK-NEXT:     Post-Dominator Tree Construction
; CHECK-NEXT:     Branch Probability Analysis
; CHECK-NEXT:     Block Frequency Analysis
; CHECK-NEXT: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT:     Natural Loop Information
; CHECK-NEXT:     Post-Dominator Tree Construction
; CHECK-NEXT:     Branch Probability Analysis
; CHECK-NEXT:     Block Frequency Analysis

; NEWPM:      VerifierPass on [module]
; NEWPM-NEXT:   VerifierAnalysis analysis on [module]
; NEWPM-NEXT: Annotation2MetadataPass on [module]
; NEWPM-NEXT: ForceFunctionAttrsPass on [module]
; NEWPM-NEXT: InferFunctionAttrsPass on [module]
; NEWPM-NEXT:   InnerAnalysisManagerProxy<{{.*}}> analysis on [module]
; NEWPM-NEXT: ModuleToFunctionPassAdaptor on [module]
; NEWPM-NEXT:   PassManager<{{.*}}> on f
; NEWPM-NEXT:     PreservedCFGCheckerAnalysis analysis on f
; NEWPM-NEXT:     LowerExpectIntrinsicPass on f
; NEWPM-NEXT:     SimplifyCFGPass on f
; NEWPM-NEXT:       TargetIRAnalysis analysis on f
; NEWPM-NEXT:       AssumptionAnalysis analysis on f
; NEWPM-NEXT:     SROA on f
; NEWPM-NEXT:       DominatorTreeAnalysis analysis on f
; NEWPM-NEXT:     EarlyCSEPass on f
; NEWPM-NEXT:       TargetLibraryAnalysis analysis on f
; NEWPM-NEXT:     CallSiteSplittingPass on f
; NEWPM-NEXT: OpenMPOptPass on [module]
; NEWPM-NEXT: IPSCCPPass on [module]
; NEWPM-NEXT: CalledValuePropagationPass on [module]
; NEWPM-NEXT: GlobalOptPass on [module]
; NEWPM-NEXT: ModuleToFunctionPassAdaptor on [module]
; NEWPM-NEXT:   InnerAnalysisManagerProxy<{{.*}}> analysis on [module]
; NEWPM-NEXT:   PromotePass on f
; NEWPM-NEXT:     PreservedCFGCheckerAnalysis analysis on f
; NEWPM-NEXT:     DominatorTreeAnalysis analysis on f
; NEWPM-NEXT:     AssumptionAnalysis analysis on f
; NEWPM-NEXT: DeadArgumentEliminationPass on [module]
; NEWPM-NEXT: ModuleToFunctionPassAdaptor on [module]
; NEWPM-NEXT:   PassManager<{{.*}}> on f
; NEWPM-NEXT:     InstCombinePass on f
; NEWPM-NEXT:       TargetLibraryAnalysis analysis on f
; NEWPM-NEXT:       OptimizationRemarkEmitterAnalysis analysis on f
; NEWPM-NEXT:       TargetIRAnalysis analysis on f
; NEWPM-NEXT:       AAManager analysis on f
; NEWPM-NEXT:         BasicAA analysis on f
; NEWPM-NEXT:       OuterAnalysisManagerProxy<{{.*}}> analysis on f
; NEWPM-NEXT:     SimplifyCFGPass on f
; NEWPM-NEXT: ModuleInlinerWrapperPass on [module]
; NEWPM-NEXT:   InlineAdvisorAnalysis analysis on [module]
; NEWPM-NEXT:   RequireAnalysisPass<{{.*}}> on [module]
; NEWPM-NEXT:     GlobalsAA analysis on [module]
; NEWPM-NEXT:       CallGraphAnalysis analysis on [module]
; NEWPM-NEXT:   ModuleToFunctionPassAdaptor on [module]
; NEWPM-NEXT:   InvalidateAnalysisPass<{{.*}}> on f
; NEWPM-NEXT:   RequireAnalysisPass<{{.*}}> on [module]
; NEWPM-NEXT:     ProfileSummaryAnalysis analysis on [module]
; NEWPM-NEXT:   ModuleToPostOrderCGSCCPassAdaptor on [module]
; NEWPM-NEXT:     InnerAnalysisManagerProxy<{{.*}}> analysis on [module]
; NEWPM-NEXT:       LazyCallGraphAnalysis analysis on [module]
; NEWPM-NEXT:     FunctionAnalysisManagerCGSCCProxy analysis on (f)
; NEWPM-NEXT:       OuterAnalysisManagerProxy<{{.*}}> analysis on (f)
; NEWPM-NEXT:     DevirtSCCRepeatedPass on (f)
; NEWPM-NEXT:       PassManager<{{.*}}> on (f)
; NEWPM-NEXT:         InlinerPass on (f)
; NEWPM-NEXT:         InlinerPass on (f)
; NEWPM-NEXT:         PostOrderFunctionAttrsPass on (f)
; NEWPM-NEXT:           AAManager analysis on f
; NEWPM-NEXT:         ArgumentPromotionPass on (f)
; NEWPM-NEXT:         OpenMPOptCGSCCPass on (f)
; NEWPM-NEXT:         CGSCCToFunctionPassAdaptor on (f)
; NEWPM-NEXT:           PassManager<{{.*}}> on f
; NEWPM-NEXT:             SROA on f
; NEWPM-NEXT:             EarlyCSEPass on f
; NEWPM-NEXT:               MemorySSAAnalysis analysis on f
; NEWPM-NEXT:             SpeculativeExecutionPass on f
; NEWPM-NEXT:             JumpThreadingPass on f
; NEWPM-NEXT:               LazyValueAnalysis analysis on f
; NEWPM-NEXT:             CorrelatedValuePropagationPass on f
; NEWPM-NEXT:             SimplifyCFGPass on f
; NEWPM-NEXT:             AggressiveInstCombinePass on f
; NEWPM-NEXT:             InstCombinePass on f
; NEWPM-NEXT:             LibCallsShrinkWrapPass on f
; NEWPM-NEXT:             TailCallElimPass on f
; NEWPM-NEXT:             SimplifyCFGPass on f
; NEWPM-NEXT:             ReassociatePass on f
; NEWPM-NEXT:             RequireAnalysisPass<{{.*}}> on f
; NEWPM-NEXT:             FunctionToLoopPassAdaptor on f
; NEWPM-NEXT:               PassManager<{{.*}}> on f
; NEWPM-NEXT:                 LoopSimplifyPass on f
; NEWPM-NEXT:                   LoopAnalysis analysis on f
; NEWPM-NEXT:                 LCSSAPass on f
; NEWPM-NEXT:             SimplifyCFGPass on f
; NEWPM-NEXT:             InstCombinePass on f
; NEWPM-NEXT:             FunctionToLoopPassAdaptor on f
; NEWPM-NEXT:               PassManager<{{.*}}> on f
; NEWPM-NEXT:                 LoopSimplifyPass on f
; NEWPM-NEXT:                 LCSSAPass on f
; NEWPM-NEXT:             SROA on f
; NEWPM-NEXT:             MergedLoadStoreMotionPass on f
; NEWPM-NEXT:             GVN on f
; NEWPM-NEXT:               MemoryDependenceAnalysis analysis on f
; NEWPM-NEXT:                 PhiValuesAnalysis analysis on f
; NEWPM-NEXT:             SCCPPass on f
; NEWPM-NEXT:             BDCEPass on f
; NEWPM-NEXT:               DemandedBitsAnalysis analysis on f
; NEWPM-NEXT:             InstCombinePass on f
; NEWPM-NEXT:             JumpThreadingPass on f
; NEWPM-NEXT:               LazyValueAnalysis analysis on f
; NEWPM-NEXT:             CorrelatedValuePropagationPass on f
; NEWPM-NEXT:             ADCEPass on f
; NEWPM-NEXT:               PostDominatorTreeAnalysis analysis on f
; NEWPM-NEXT:             MemCpyOptPass on f
; NEWPM-NEXT:             DSEPass on f
; NEWPM-NEXT:             FunctionToLoopPassAdaptor on f
; NEWPM-NEXT:               PassManager<{{.*}}> on f
; NEWPM-NEXT:                 LoopSimplifyPass on f
; NEWPM-NEXT:                 LCSSAPass on f
; NEWPM-NEXT:             SimplifyCFGPass on f
; NEWPM-NEXT:             InstCombinePass on f
; NEWPM-NEXT: GlobalOptPass on [module]
; NEWPM-NEXT: GlobalDCEPass on [module]
; NEWPM-NEXT: EliminateAvailableExternallyPass on [module]
; NEWPM-NEXT: ReversePostOrderFunctionAttrsPass on [module]
; NEWPM-NEXT:   CallGraphAnalysis analysis on [module]
; NEWPM-NEXT: RequireAnalysisPass<{{.*}}> on [module]
; NEWPM-NEXT: ModuleToFunctionPassAdaptor on [module]
; NEWPM-NEXT:   PassManager<{{.*}}> on f
; NEWPM-NEXT:     Float2IntPass on f
; NEWPM-NEXT:     LowerConstantIntrinsicsPass on f
; NEWPM-NEXT:     FunctionToLoopPassAdaptor on f
; NEWPM-NEXT:       PassManager<{{.*}}> on f
; NEWPM-NEXT:         LoopSimplifyPass on f
; NEWPM-NEXT:         LCSSAPass on f
; NEWPM-NEXT:     LoopDistributePass on f
; NEWPM-NEXT:       ScalarEvolutionAnalysis analysis on f
; NEWPM-NEXT:       InnerAnalysisManagerProxy<{{.*}}> analysis on f
; NEWPM-NEXT:     InjectTLIMappings on f
; NEWPM-NEXT:     LoopVectorizePass on f
; NEWPM-NEXT:       BlockFrequencyAnalysis analysis on f
; NEWPM-NEXT:         BranchProbabilityAnalysis analysis on f
; NEWPM-NEXT:     LoopLoadEliminationPass on f
; NEWPM-NEXT:     InstCombinePass on f
; NEWPM-NEXT:     SimplifyCFGPass on f
; NEWPM-NEXT:     SLPVectorizerPass on f
; NEWPM-NEXT:     VectorCombinePass on f
; NEWPM-NEXT:     InstCombinePass on f
; NEWPM-NEXT:     LoopUnrollPass on f
; NEWPM-NEXT:     WarnMissedTransformationsPass on f
; NEWPM-NEXT:     InstCombinePass on f
; NEWPM-NEXT:     RequireAnalysisPass<{{.*}}> on f
; NEWPM-NEXT:     FunctionToLoopPassAdaptor on f
; NEWPM-NEXT:       PassManager<{{.*}}> on f
; NEWPM-NEXT:         LoopSimplifyPass on f
; NEWPM-NEXT:         LCSSAPass on f
; NEWPM-NEXT:     AlignmentFromAssumptionsPass on f
; NEWPM-NEXT:     LoopSinkPass on f
; NEWPM-NEXT:     InstSimplifyPass on f
; NEWPM-NEXT:     DivRemPairsPass on f
; NEWPM-NEXT:     SimplifyCFGPass on f
; NEWPM-NEXT:     SpeculateAroundPHIsPass on f
; NEWPM-NEXT: CGProfilePass on [module]
; NEWPM-NEXT: GlobalDCEPass on [module]
; NEWPM-NEXT: ConstantMergePass on [module]
; NEWPM-NEXT: RelLookupTableConverterPass on [module]
; NEWPM-NEXT: ModuleToFunctionPassAdaptor on [module]
; NEWPM-NEXT:   PassManager<{{.*}}> on f
; NEWPM-NEXT:     AnnotationRemarksPass on f
; NEWPM-NEXT: VerifierPass on [module]
; NEWPM-NEXT:   VerifierAnalysis analysis on [module]
; NEWPM-NEXT: BitcodeWriterPass on [module]

define void @f() {
  ret void
}
