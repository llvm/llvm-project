; RUN: opt -mtriple=x86_64-- -O3 -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck %s

; REQUIRES: asserts

; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Transform Information
; CHECK-NEXT: Type-Based Alias Analysis
; CHECK-NEXT: Scoped NoAlias Alias Analysis
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-NEXT: Target Library Information
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Module Verifier
; CHECK-NEXT:     Instrument function entry/exit with calls to e.g. mcount() (pre inlining)
; CHECK-NEXT:     Simplify the CFG
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT:     SROA
; CHECK-NEXT:     Early CSE
; CHECK-NEXT:     Lower 'expect' Intrinsics
; CHECK-NEXT: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: Target Transform Information
;             Target Pass Configuration
; CHECK:      Type-Based Alias Analysis
; CHECK-NEXT: Scoped NoAlias Alias Analysis
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-NEXT: Profile summary info
; CHECK-NEXT:   ModulePass Manager
; CHECK-NEXT:     Force set function attributes
; CHECK-NEXT:     Infer set function attributes
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Call-site splitting
; CHECK-NEXT:     Interprocedural Sparse Conditional Constant Propagation
; CHECK-NEXT:       Unnamed pass: implement Pass::getPassName()
; CHECK-NEXT:     Called Value Propagation
; CHECK-NEXT:     Global Variable Optimizer
; CHECK-NEXT:       Unnamed pass: implement Pass::getPassName()
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
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
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
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Canonicalize natural loops
; CHECK-NEXT:         LCSSA Verifier
; CHECK-NEXT:         Loop-Closed SSA Form Pass
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Scalar Evolution Analysis
; CHECK-NEXT:         Loop Pass Manager
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
; CHECK-NEXT:           Induction Variable Simplification
; CHECK-NEXT:           Recognize loop idioms
; CHECK-NEXT:           Delete dead loops
; CHECK-NEXT:           Unroll loops
; CHECK-NEXT:         MergedLoadStoreMotion
; CHECK-NEXT:         Phi Values Analysis
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Memory Dependence Analysis
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:         Optimization Remark Emitter
; CHECK-NEXT:         Global Value Numbering
; CHECK-NEXT:         Phi Values Analysis
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Memory Dependence Analysis
; CHECK-NEXT:         MemCpy Optimization
; CHECK-NEXT:         Sparse Conditional Constant Propagation
; CHECK-NEXT:         Demanded bits analysis
; CHECK-NEXT:         Bit-Tracking Dead Code Elimination
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Lazy Branch Probability Analysis
; CHECK-NEXT:         Lazy Block Frequency Analysis
; CHECK-NEXT:         Optimization Remark Emitter
; CHECK-NEXT:         Combine redundant instructions
; CHECK-NEXT:         Lazy Value Information Analysis
; CHECK-NEXT:         Jump Threading
; CHECK-NEXT:         Value Propagation
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Phi Values Analysis
; CHECK-NEXT:         Memory Dependence Analysis
; CHECK-NEXT:         Dead Store Elimination
; CHECK-NEXT:         Natural Loop Information
; CHECK-NEXT:         Canonicalize natural loops
; CHECK-NEXT:         LCSSA Verifier
; CHECK-NEXT:         Loop-Closed SSA Form Pass
; CHECK-NEXT:         Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:         Function Alias Analysis Results
; CHECK-NEXT:         Scalar Evolution Analysis
; CHECK-NEXT:         Loop Pass Manager
; CHECK-NEXT:           Loop Invariant Code Motion
; CHECK-NEXT:         Post-Dominator Tree Construction
; CHECK-NEXT:         Aggressive Dead Code Elimination
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
; CHECK-NEXT:       Unnamed pass: implement Pass::getPassName()
; CHECK-NEXT:     Dead Global Elimination
; CHECK-NEXT:     CallGraph Construction
; CHECK-NEXT:     Globals Alias Analysis
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Float to int
; CHECK-NEXT:       Dominator Tree Construction
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
; CHECK-NEXT:       Loop Vectorization
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Loop Access Analysis
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
; CHECK-NEXT:       SLP Vectorizer
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
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       LCSSA Verifier
; CHECK-NEXT:       Loop-Closed SSA Form Pass
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Loop Pass Manager
; CHECK-NEXT:         Loop Invariant Code Motion
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Warn about non-applied transformations
; CHECK-NEXT:       Alignment from assumptions
; CHECK-NEXT:     Strip Unused Function Prototypes
; CHECK-NEXT:     Dead Global Elimination
; CHECK-NEXT:     Merge Duplicate Global Constants
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Branch Probability Analysis
; CHECK-NEXT:       Block Frequency Analysis
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       LCSSA Verifier
; CHECK-NEXT:       Loop-Closed SSA Form Pass
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Branch Probability Analysis
; CHECK-NEXT:       Block Frequency Analysis
; CHECK-NEXT:       Loop Pass Manager
; CHECK-NEXT:         Loop Sink
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       Remove redundant instructions
; CHECK-NEXT:       Hoist/decompose integer division and remainder
; CHECK-NEXT:       Simplify the CFG
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
; CHECK-NEXT:     Branch Probability Analysis
; CHECK-NEXT:     Block Frequency Analysis
; CHECK-NEXT: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT:     Natural Loop Information
; CHECK-NEXT:     Branch Probability Analysis
; CHECK-NEXT:     Block Frequency Analysis

define void @f() {
  ret void
}
