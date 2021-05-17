; RUN: opt -enable-new-pm=0 -mtriple=x86_64-- -std-link-opts -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK %s

; REQUIRES: asserts

; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: Target Transform Information
;           : Target Pass Configuration
; CHECK:      Type-Based Alias Analysis
; CHECK-NEXT: Scoped NoAlias Alias Analysis
; CHECK-NEXT: Profile summary info
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-NEXT:   ModulePass Manager
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:     Dead Global Elimination
; CHECK-NEXT:     Force set function attributes
; CHECK-NEXT:     Infer set function attributes
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Call-site splitting
; CHECK-NEXT:     PGOIndirectCallPromotion
; CHECK-NEXT:     Interprocedural Sparse Conditional Constant Propagation
; CHECK-NEXT:       FunctionPass Manager
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:     Called Value Propagation
; CHECK-NEXT:     CallGraph Construction
; CHECK-NEXT:     Call Graph SCC Pass Manager
; CHECK-NEXT:       Deduce function attributes
; CHECK-NEXT:     Deduce function attributes in RPO
; CHECK-NEXT:     Global splitter
; CHECK-NEXT:     Whole program devirtualization
; CHECK-NEXT:       FunctionPass Manager
; CHECK-NEXT:         Dominator Tree Construction
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
; CHECK-NEXT:     Merge Duplicate Global Constants
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
; CHECK-NEXT:     CallGraph Construction
; CHECK-NEXT:     Call Graph SCC Pass Manager
; CHECK-NEXT:       Function Integration/Inlining
; CHECK-NEXT:       Remove unused exception handling info
; CHECK-NEXT:       OpenMP specific optimizations
; CHECK-NEXT:     Global Variable Optimizer
; CHECK-NEXT:       FunctionPass Manager
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Post-Dominator Tree Construction
; CHECK-NEXT:         Branch Probability Analysis
; CHECK-NEXT:         Block Frequency Analysis
; CHECK-NEXT:     Dead Global Elimination
; CHECK-NEXT:     CallGraph Construction
; CHECK-NEXT:     Call Graph SCC Pass Manager
; CHECK-NEXT:       Promote 'by reference' arguments to scalars
; CHECK-NEXT:       FunctionPass Manager
; CHECK-NEXT:         Dominator Tree Construction
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:         Optimization Remark Emitter
; CHECK-NEXT:         Combine redundant instructions
; CHECK-NEXT:         Lazy Value Information Analysis
; CHECK-NEXT:         Jump Threading
; CHECK-NEXT:         SROA
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:         Optimization Remark Emitter
; CHECK-NEXT:         Tail Call Elimination
; CHECK-NEXT:     CallGraph Construction
; CHECK-NEXT:     Call Graph SCC Pass Manager
; CHECK-NEXT:       Deduce function attributes
; CHECK-NEXT:     Globals Alias Analysis
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Memory SSA
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       LCSSA Verifier
; CHECK-NEXT:       Loop-Closed SSA Form Pass
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Loop Pass Manager
; CHECK-NEXT:         Loop Invariant Code Motion
; CHECK-NEXT:       Phi Values Analysis
; CHECK-NEXT:       Memory Dependence Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Global Value Numbering
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       MemCpy Optimization
; CHECK-NEXT:       Post-Dominator Tree Construction
; CHECK-NEXT:       Dead Store Elimination
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       MergedLoadStoreMotion
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       LCSSA Verifier
; CHECK-NEXT:       Loop-Closed SSA Form Pass
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Loop Pass Manager
; CHECK-NEXT:         Induction Variable Simplification
; CHECK-NEXT:         Delete dead loops
; CHECK-NEXT:         Unroll loops
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
; CHECK-NEXT:       LCSSA Verifier
; CHECK-NEXT:       Loop-Closed SSA Form Pass
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Loop Pass Manager
; CHECK-NEXT:         Unroll loops
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Warn about non-applied transformations
; CHECK-NEXT:       Combine redundant instructions
; CHECK-NEXT:       Simplify the CFG
; CHECK-NEXT:       Sparse Conditional Constant Propagation
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Combine redundant instructions
; CHECK-NEXT:       Demanded bits analysis
; CHECK-NEXT:       Bit-Tracking Dead Code Elimination
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Optimize scalar/vector ops
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Alignment from assumptions
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Combine redundant instructions
; CHECK-NEXT:       Lazy Value Information Analysis
; CHECK-NEXT:       Jump Threading
; CHECK-NEXT:     Cross-DSO CFI
; CHECK-NEXT:     Lower type metadata
; CHECK-NEXT:     Lower type metadata
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Simplify the CFG
; CHECK-NEXT:     Eliminate Available Externally Globals
; CHECK-NEXT:     Dead Global Elimination
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Annotation Remarks
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:     Bitcode Writer
; CHECK-NEXT: Pass Arguments:
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT: Pass Arguments:
; CHECK-NEXT:   FunctionPass Manager
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

define void @f() {
  ret void
}
