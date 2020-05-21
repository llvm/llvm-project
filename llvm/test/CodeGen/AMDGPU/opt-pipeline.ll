; RUN: opt -O0 -mtriple=amdgcn--amdhsa -disable-output -disable-verify -debug-pass=Structure %s 2>&1 | FileCheck -check-prefix=GCN-O0 %s
; RUN: opt -O1 -mtriple=amdgcn--amdhsa -disable-output -disable-verify -debug-pass=Structure %s 2>&1 | FileCheck -check-prefix=GCN-O1 %s
; RUN: opt -O2 -mtriple=amdgcn--amdhsa -disable-output -disable-verify -debug-pass=Structure %s 2>&1 | FileCheck -check-prefix=GCN-O2 %s
; RUN: opt -O3 -mtriple=amdgcn--amdhsa -disable-output -disable-verify -debug-pass=Structure %s 2>&1 | FileCheck -check-prefix=GCN-O3 %s

; REQUIRES: asserts

; GCN-O0:      Pass Arguments:
; GCN-O0-NEXT: Target Transform Information
; GCN-O0-NEXT:   FunctionPass Manager
; GCN-O0-NEXT:     Early propagate attributes from kernels to functions
; GCN-O0-NEXT:     Replace builtin math calls with that native versions.
; GCN-O0-NEXT:     Instrument function entry/exit with calls to e.g. mcount() (pre inlining)

; GCN-O0-NEXT: Pass Arguments:
; GCN-O0-NEXT: Target Library Information
; GCN-O0-NEXT: Target Transform Information
; GCN-O0-NEXT: Target Pass Configuration
; GCN-O0-NEXT: Assumption Cache Tracker
; GCN-O0-NEXT: Profile summary info
; GCN-O0-NEXT:   ModulePass Manager
; GCN-O0-NEXT:     Force set function attributes
; GCN-O0-NEXT:     CallGraph Construction
; GCN-O0-NEXT:     Call Graph SCC Pass Manager
; GCN-O0-NEXT:       AMDGPU Function Integration/Inlining
; GCN-O0-NEXT:     A No-Op Barrier Pass


; GCN-O1:      Pass Arguments:
; GCN-O1-NEXT: Target Transform Information
; GCN-O1-NEXT: AMDGPU Address space based Alias Analysis
; GCN-O1-NEXT: External Alias Analysis
; GCN-O1-NEXT: Assumption Cache Tracker
; GCN-O1-NEXT: Target Library Information
; GCN-O1-NEXT: Type-Based Alias Analysis
; GCN-O1-NEXT: Scoped NoAlias Alias Analysis
; GCN-O1-NEXT:   FunctionPass Manager
; GCN-O1-NEXT:     Early propagate attributes from kernels to functions
; GCN-O1-NEXT:     Replace builtin math calls with that native versions.
; GCN-O1-NEXT:     Dominator Tree Construction
; GCN-O1-NEXT:     Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:     Function Alias Analysis Results
; GCN-O1-NEXT:     Simplify well-known AMD library calls
; GCN-O1-NEXT:     Instrument function entry/exit with calls to e.g. mcount() (pre inlining)
; GCN-O1-NEXT:     Simplify the CFG
; GCN-O1-NEXT:     Dominator Tree Construction
; GCN-O1-NEXT:     SROA
; GCN-O1-NEXT:     Early CSE
; GCN-O1-NEXT:     Lower 'expect' Intrinsics

; GCN-O1-NEXT: Pass Arguments:
; GCN-O1-NEXT: Target Library Information
; GCN-O1-NEXT: Target Transform Information
; GCN-O1-NEXT: Target Pass Configuration
; GCN-O1-NEXT: Type-Based Alias Analysis
; GCN-O1-NEXT: Scoped NoAlias Alias Analysis
; GCN-O1-NEXT: AMDGPU Address space based Alias Analysis
; GCN-O1-NEXT: External Alias Analysis
; GCN-O1-NEXT: Assumption Cache Tracker
; GCN-O1-NEXT: Profile summary info
; GCN-O1-NEXT:   ModulePass Manager
; GCN-O1-NEXT:     Force set function attributes
; GCN-O1-NEXT:     Infer set function attributes
; GCN-O1-NEXT:     Unify multiple OpenCL metadata due to linking
; GCN-O1-NEXT:     AMDGPU Printf lowering
; GCN-O1-NEXT:       FunctionPass Manager
; GCN-O1-NEXT:         Dominator Tree Construction
; GCN-O1-NEXT:     Late propagate attributes from kernels to functions
; GCN-O1-NEXT:     Interprocedural Sparse Conditional Constant Propagation
; GCN-O1-NEXT:       FunctionPass Manager
; GCN-O1-NEXT:         Dominator Tree Construction
; GCN-O1-NEXT:     Called Value Propagation
; GCN-O1-NEXT:     Global Variable Optimizer
; GCN-O1-NEXT:       FunctionPass Manager
; GCN-O1-NEXT:         Dominator Tree Construction
; GCN-O1-NEXT:         Natural Loop Information
; GCN-O1-NEXT:         Post-Dominator Tree Construction
; GCN-O1-NEXT:         Branch Probability Analysis
; GCN-O1-NEXT:         Block Frequency Analysis
; GCN-O1-NEXT:     FunctionPass Manager
; GCN-O1-NEXT:       Dominator Tree Construction
; GCN-O1-NEXT:       Promote Memory to Register
; GCN-O1-NEXT:     Dead Argument Elimination
; GCN-O1-NEXT:     FunctionPass Manager
; GCN-O1-NEXT:       Dominator Tree Construction
; GCN-O1-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:       Function Alias Analysis Results
; GCN-O1-NEXT:       Natural Loop Information
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Optimization Remark Emitter
; GCN-O1-NEXT:       Combine redundant instructions
; GCN-O1-NEXT:       Simplify the CFG
; GCN-O1-NEXT:     CallGraph Construction
; GCN-O1-NEXT:     Globals Alias Analysis
; GCN-O1-NEXT:     Call Graph SCC Pass Manager
; GCN-O1-NEXT:       Remove unused exception handling info
; GCN-O1-NEXT:       AMDGPU Function Integration/Inlining
; GCN-O1-NEXT:       Deduce function attributes
; GCN-O1-NEXT:       FunctionPass Manager
; GCN-O1-NEXT:         Infer address spaces
; GCN-O1-NEXT:     AMDGPU Kernel Attributes
; GCN-O1-NEXT:     FunctionPass Manager
; GCN-O1-NEXT:       AMDGPU Promote Alloca to vector
; GCN-O1-NEXT:       Dominator Tree Construction
; GCN-O1-NEXT:       SROA
; GCN-O1-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:       Function Alias Analysis Results
; GCN-O1-NEXT:       Memory SSA
; GCN-O1-NEXT:       Early CSE w/ MemorySSA
; GCN-O1-NEXT:       Simplify the CFG
; GCN-O1-NEXT:       Dominator Tree Construction
; GCN-O1-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:       Function Alias Analysis Results
; GCN-O1-NEXT:       Natural Loop Information
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Optimization Remark Emitter
; GCN-O1-NEXT:       Combine redundant instructions
; GCN-O1-NEXT:       Conditionally eliminate dead library calls
; GCN-O1-NEXT:       Natural Loop Information
; GCN-O1-NEXT:       Post-Dominator Tree Construction
; GCN-O1-NEXT:       Branch Probability Analysis
; GCN-O1-NEXT:       Block Frequency Analysis
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Optimization Remark Emitter
; GCN-O1-NEXT:       PGOMemOPSize
; GCN-O1-NEXT:       Simplify the CFG
; GCN-O1-NEXT:       Reassociate expressions
; GCN-O1-NEXT:       Dominator Tree Construction
; GCN-O1-NEXT:       Natural Loop Information
; GCN-O1-NEXT:       Canonicalize natural loops
; GCN-O1-NEXT:       LCSSA Verifier
; GCN-O1-NEXT:       Loop-Closed SSA Form Pass
; GCN-O1-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:       Function Alias Analysis Results
; GCN-O1-NEXT:       Scalar Evolution Analysis
; GCN-O1-NEXT:       Loop Pass Manager
; GCN-O1-NEXT:         Rotate Loops
; GCN-O1-NEXT:       Memory SSA
; GCN-O1-NEXT:       Loop Pass Manager
; GCN-O1-NEXT:         Loop Invariant Code Motion
; GCN-O1-NEXT:       Post-Dominator Tree Construction
; GCN-O1-NEXT:       Legacy Divergence Analysis
; GCN-O1-NEXT:       Loop Pass Manager
; GCN-O1-NEXT:         Unswitch loops
; GCN-O1-NEXT:       Simplify the CFG
; GCN-O1-NEXT:       Dominator Tree Construction
; GCN-O1-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:       Function Alias Analysis Results
; GCN-O1-NEXT:       Natural Loop Information
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Optimization Remark Emitter
; GCN-O1-NEXT:       Combine redundant instructions
; GCN-O1-NEXT:       Canonicalize natural loops
; GCN-O1-NEXT:       LCSSA Verifier
; GCN-O1-NEXT:       Loop-Closed SSA Form Pass
; GCN-O1-NEXT:       Scalar Evolution Analysis
; GCN-O1-NEXT:       Loop Pass Manager
; GCN-O1-NEXT:         Induction Variable Simplification
; GCN-O1-NEXT:         Recognize loop idioms
; GCN-O1-NEXT:         Delete dead loops
; GCN-O1-NEXT:         Unroll loops
; GCN-O1-NEXT:       Phi Values Analysis
; GCN-O1-NEXT:       Memory Dependence Analysis
; GCN-O1-NEXT:       MemCpy Optimization
; GCN-O1-NEXT:       Sparse Conditional Constant Propagation
; GCN-O1-NEXT:       Demanded bits analysis
; GCN-O1-NEXT:       Bit-Tracking Dead Code Elimination
; GCN-O1-NEXT:       Function Alias Analysis Results
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Optimization Remark Emitter
; GCN-O1-NEXT:       Combine redundant instructions
; GCN-O1-NEXT:       Post-Dominator Tree Construction
; GCN-O1-NEXT:       Aggressive Dead Code Elimination
; GCN-O1-NEXT:       Simplify the CFG
; GCN-O1-NEXT:       Dominator Tree Construction
; GCN-O1-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:       Function Alias Analysis Results
; GCN-O1-NEXT:       Natural Loop Information
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Optimization Remark Emitter
; GCN-O1-NEXT:       Combine redundant instructions
; GCN-O1-NEXT:     A No-Op Barrier Pass
; GCN-O1-NEXT:     CallGraph Construction
; GCN-O1-NEXT:     Deduce function attributes in RPO
; GCN-O1-NEXT:     Global Variable Optimizer
; GCN-O1-NEXT:       FunctionPass Manager
; GCN-O1-NEXT:         Dominator Tree Construction
; GCN-O1-NEXT:         Natural Loop Information
; GCN-O1-NEXT:         Post-Dominator Tree Construction
; GCN-O1-NEXT:         Branch Probability Analysis
; GCN-O1-NEXT:         Block Frequency Analysis
; GCN-O1-NEXT:     Dead Global Elimination
; GCN-O1-NEXT:     CallGraph Construction
; GCN-O1-NEXT:     Globals Alias Analysis
; GCN-O1-NEXT:     FunctionPass Manager
; GCN-O1-NEXT:       Dominator Tree Construction
; GCN-O1-NEXT:       Float to int
; GCN-O1-NEXT:       Lower constant intrinsics
; GCN-O1-NEXT:       Dominator Tree Construction
; GCN-O1-NEXT:       Natural Loop Information
; GCN-O1-NEXT:       Canonicalize natural loops
; GCN-O1-NEXT:       LCSSA Verifier
; GCN-O1-NEXT:       Loop-Closed SSA Form Pass
; GCN-O1-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:       Function Alias Analysis Results
; GCN-O1-NEXT:       Scalar Evolution Analysis
; GCN-O1-NEXT:       Loop Pass Manager
; GCN-O1-NEXT:         Rotate Loops
; GCN-O1-NEXT:       Loop Access Analysis
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Optimization Remark Emitter
; GCN-O1-NEXT:       Loop Distribution
; GCN-O1-NEXT:       Post-Dominator Tree Construction
; GCN-O1-NEXT:       Branch Probability Analysis
; GCN-O1-NEXT:       Block Frequency Analysis
; GCN-O1-NEXT:       Scalar Evolution Analysis
; GCN-O1-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:       Function Alias Analysis Results
; GCN-O1-NEXT:       Loop Access Analysis
; GCN-O1-NEXT:       Demanded bits analysis
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Optimization Remark Emitter
; GCN-O1-NEXT:       Inject TLI Mappings
; GCN-O1-NEXT:       Loop Vectorization
; GCN-O1-NEXT:       Optimize scalar/vector ops
; GCN-O1-NEXT:       Early CSE
; GCN-O1-NEXT:       Canonicalize natural loops
; GCN-O1-NEXT:       Scalar Evolution Analysis
; GCN-O1-NEXT:       Function Alias Analysis Results
; GCN-O1-NEXT:       Loop Access Analysis
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Loop Load Elimination
; GCN-O1-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:       Function Alias Analysis Results
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Optimization Remark Emitter
; GCN-O1-NEXT:       Combine redundant instructions
; GCN-O1-NEXT:       Simplify the CFG
; GCN-O1-NEXT:       Dominator Tree Construction
; GCN-O1-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:       Function Alias Analysis Results
; GCN-O1-NEXT:       Natural Loop Information
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Optimization Remark Emitter
; GCN-O1-NEXT:       Combine redundant instructions
; GCN-O1-NEXT:       Canonicalize natural loops
; GCN-O1-NEXT:       LCSSA Verifier
; GCN-O1-NEXT:       Loop-Closed SSA Form Pass
; GCN-O1-NEXT:       Scalar Evolution Analysis
; GCN-O1-NEXT:       Loop Pass Manager
; GCN-O1-NEXT:         Unroll loops
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Optimization Remark Emitter
; GCN-O1-NEXT:       Combine redundant instructions
; GCN-O1-NEXT:       Memory SSA
; GCN-O1-NEXT:       Canonicalize natural loops
; GCN-O1-NEXT:       LCSSA Verifier
; GCN-O1-NEXT:       Loop-Closed SSA Form Pass
; GCN-O1-NEXT:       Scalar Evolution Analysis
; GCN-O1-NEXT:       Loop Pass Manager
; GCN-O1-NEXT:         Loop Invariant Code Motion
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Optimization Remark Emitter
; GCN-O1-NEXT:       Warn about non-applied transformations
; GCN-O1-NEXT:       Alignment from assumptions
; GCN-O1-NEXT:     Strip Unused Function Prototypes
; GCN-O1-NEXT:     FunctionPass Manager
; GCN-O1-NEXT:       Dominator Tree Construction
; GCN-O1-NEXT:       Natural Loop Information
; GCN-O1-NEXT:       Post-Dominator Tree Construction
; GCN-O1-NEXT:       Branch Probability Analysis
; GCN-O1-NEXT:       Block Frequency Analysis
; GCN-O1-NEXT:       Canonicalize natural loops
; GCN-O1-NEXT:       LCSSA Verifier
; GCN-O1-NEXT:       Loop-Closed SSA Form Pass
; GCN-O1-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:       Function Alias Analysis Results
; GCN-O1-NEXT:       Scalar Evolution Analysis
; GCN-O1-NEXT:       Block Frequency Analysis
; GCN-O1-NEXT:       Loop Pass Manager
; GCN-O1-NEXT:         Loop Sink
; GCN-O1-NEXT:       Lazy Branch Probability Analysis
; GCN-O1-NEXT:       Lazy Block Frequency Analysis
; GCN-O1-NEXT:       Optimization Remark Emitter
; GCN-O1-NEXT:       Remove redundant instructions
; GCN-O1-NEXT:       Hoist/decompose integer division and remainder
; GCN-O1-NEXT:       Simplify the CFG

; GCN-O1-NEXT: Pass Arguments:
; GCN-O1-NEXT:   FunctionPass Manager
; GCN-O1-NEXT:     Dominator Tree Construction

; GCN-O1-NEXT: Pass Arguments:
; GCN-O1-NEXT:   FunctionPass Manager
; GCN-O1-NEXT:     Dominator Tree Construction

; GCN-O1-NEXT: Pass Arguments:
; GCN-O1-NEXT: Target Library Information
; GCN-O1-NEXT:   FunctionPass Manager
; GCN-O1-NEXT:     Dominator Tree Construction
; GCN-O1-NEXT:     Natural Loop Information
; GCN-O1-NEXT:     Post-Dominator Tree Construction
; GCN-O1-NEXT:     Branch Probability Analysis
; GCN-O1-NEXT:     Block Frequency Analysis

; GCN-O1-NEXT: Pass Arguments:
; GCN-O1-NEXT: Target Library Information
; GCN-O1-NEXT:   FunctionPass Manager
; GCN-O1-NEXT:     Dominator Tree Construction
; GCN-O1-NEXT:     Natural Loop Information
; GCN-O1-NEXT:     Post-Dominator Tree Construction
; GCN-O1-NEXT:     Branch Probability Analysis
; GCN-O1-NEXT:     Block Frequency Analysis


; GCN-O2:      Pass Arguments:
; GCN-O2-NEXT: Target Transform Information
; GCN-O2-NEXT: AMDGPU Address space based Alias Analysis
; GCN-O2-NEXT: External Alias Analysis
; GCN-O2-NEXT: Assumption Cache Tracker
; GCN-O2-NEXT: Target Library Information
; GCN-O2-NEXT: Type-Based Alias Analysis
; GCN-O2-NEXT: Scoped NoAlias Alias Analysis
; GCN-O2-NEXT:   FunctionPass Manager
; GCN-O2-NEXT:     Early propagate attributes from kernels to functions
; GCN-O2-NEXT:     Replace builtin math calls with that native versions.
; GCN-O2-NEXT:     Dominator Tree Construction
; GCN-O2-NEXT:     Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:     Function Alias Analysis Results
; GCN-O2-NEXT:     Simplify well-known AMD library calls
; GCN-O2-NEXT:     Instrument function entry/exit with calls to e.g. mcount() (pre inlining)
; GCN-O2-NEXT:     Simplify the CFG
; GCN-O2-NEXT:     Dominator Tree Construction
; GCN-O2-NEXT:     SROA
; GCN-O2-NEXT:     Early CSE
; GCN-O2-NEXT:     Lower 'expect' Intrinsics

; GCN-O2-NEXT: Pass Arguments:
; GCN-O2-NEXT: Target Library Information
; GCN-O2-NEXT: Target Transform Information
; GCN-O2-NEXT: Target Pass Configuration
; GCN-O2-NEXT: Type-Based Alias Analysis
; GCN-O2-NEXT: Scoped NoAlias Alias Analysis
; GCN-O2-NEXT: AMDGPU Address space based Alias Analysis
; GCN-O2-NEXT: External Alias Analysis
; GCN-O2-NEXT: Assumption Cache Tracker
; GCN-O2-NEXT: Profile summary info
; GCN-O2-NEXT:   ModulePass Manager
; GCN-O2-NEXT:     Force set function attributes
; GCN-O2-NEXT:     Infer set function attributes
; GCN-O2-NEXT:     Unify multiple OpenCL metadata due to linking
; GCN-O2-NEXT:     AMDGPU Printf lowering
; GCN-O2-NEXT:       FunctionPass Manager
; GCN-O2-NEXT:         Dominator Tree Construction
; GCN-O2-NEXT:     Late propagate attributes from kernels to functions
; GCN-O2-NEXT:     Interprocedural Sparse Conditional Constant Propagation
; GCN-O2-NEXT:       FunctionPass Manager
; GCN-O2-NEXT:         Dominator Tree Construction
; GCN-O2-NEXT:     Called Value Propagation
; GCN-O2-NEXT:     Global Variable Optimizer
; GCN-O2-NEXT:       FunctionPass Manager
; GCN-O2-NEXT:         Dominator Tree Construction
; GCN-O2-NEXT:         Natural Loop Information
; GCN-O2-NEXT:         Post-Dominator Tree Construction
; GCN-O2-NEXT:         Branch Probability Analysis
; GCN-O2-NEXT:         Block Frequency Analysis
; GCN-O2-NEXT:     FunctionPass Manager
; GCN-O2-NEXT:       Dominator Tree Construction
; GCN-O2-NEXT:       Promote Memory to Register
; GCN-O2-NEXT:     Dead Argument Elimination
; GCN-O2-NEXT:     FunctionPass Manager
; GCN-O2-NEXT:       Dominator Tree Construction
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Natural Loop Information
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Combine redundant instructions
; GCN-O2-NEXT:       Simplify the CFG
; GCN-O2-NEXT:     CallGraph Construction
; GCN-O2-NEXT:     Globals Alias Analysis
; GCN-O2-NEXT:     Call Graph SCC Pass Manager
; GCN-O2-NEXT:       Remove unused exception handling info
; GCN-O2-NEXT:       AMDGPU Function Integration/Inlining
; GCN-O2-NEXT:       OpenMP specific optimizations
; GCN-O2-NEXT:       Deduce function attributes
; GCN-O2-NEXT:       FunctionPass Manager
; GCN-O2-NEXT:         Infer address spaces
; GCN-O2-NEXT:     AMDGPU Kernel Attributes
; GCN-O2-NEXT:     FunctionPass Manager
; GCN-O2-NEXT:       AMDGPU Promote Alloca to vector
; GCN-O2-NEXT:       Dominator Tree Construction
; GCN-O2-NEXT:       SROA
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Memory SSA
; GCN-O2-NEXT:       Early CSE w/ MemorySSA
; GCN-O2-NEXT:       Speculatively execute instructions if target has divergent branches
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Lazy Value Information Analysis
; GCN-O2-NEXT:       Jump Threading
; GCN-O2-NEXT:       Value Propagation
; GCN-O2-NEXT:       Simplify the CFG
; GCN-O2-NEXT:       Dominator Tree Construction
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Natural Loop Information
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Combine redundant instructions
; GCN-O2-NEXT:       Conditionally eliminate dead library calls
; GCN-O2-NEXT:       Natural Loop Information
; GCN-O2-NEXT:       Post-Dominator Tree Construction
; GCN-O2-NEXT:       Branch Probability Analysis
; GCN-O2-NEXT:       Block Frequency Analysis
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       PGOMemOPSize
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Natural Loop Information
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Tail Call Elimination
; GCN-O2-NEXT:       Simplify the CFG
; GCN-O2-NEXT:       Reassociate expressions
; GCN-O2-NEXT:       Dominator Tree Construction
; GCN-O2-NEXT:       Natural Loop Information
; GCN-O2-NEXT:       Canonicalize natural loops
; GCN-O2-NEXT:       LCSSA Verifier
; GCN-O2-NEXT:       Loop-Closed SSA Form Pass
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Scalar Evolution Analysis
; GCN-O2-NEXT:       Loop Pass Manager
; GCN-O2-NEXT:         Rotate Loops
; GCN-O2-NEXT:       Memory SSA
; GCN-O2-NEXT:       Loop Pass Manager
; GCN-O2-NEXT:         Loop Invariant Code Motion
; GCN-O2-NEXT:       Post-Dominator Tree Construction
; GCN-O2-NEXT:       Legacy Divergence Analysis
; GCN-O2-NEXT:       Loop Pass Manager
; GCN-O2-NEXT:         Unswitch loops
; GCN-O2-NEXT:       Simplify the CFG
; GCN-O2-NEXT:       Dominator Tree Construction
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Natural Loop Information
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Combine redundant instructions
; GCN-O2-NEXT:       Canonicalize natural loops
; GCN-O2-NEXT:       LCSSA Verifier
; GCN-O2-NEXT:       Loop-Closed SSA Form Pass
; GCN-O2-NEXT:       Scalar Evolution Analysis
; GCN-O2-NEXT:       Loop Pass Manager
; GCN-O2-NEXT:         Induction Variable Simplification
; GCN-O2-NEXT:         Recognize loop idioms
; GCN-O2-NEXT:         Delete dead loops
; GCN-O2-NEXT:         Unroll loops
; GCN-O2-NEXT:       MergedLoadStoreMotion
; GCN-O2-NEXT:       Phi Values Analysis
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Memory Dependence Analysis
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Global Value Numbering
; GCN-O2-NEXT:       Phi Values Analysis
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Memory Dependence Analysis
; GCN-O2-NEXT:       MemCpy Optimization
; GCN-O2-NEXT:       Sparse Conditional Constant Propagation
; GCN-O2-NEXT:       Demanded bits analysis
; GCN-O2-NEXT:       Bit-Tracking Dead Code Elimination
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Combine redundant instructions
; GCN-O2-NEXT:       Lazy Value Information Analysis
; GCN-O2-NEXT:       Jump Threading
; GCN-O2-NEXT:       Value Propagation
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Phi Values Analysis
; GCN-O2-NEXT:       Memory Dependence Analysis
; GCN-O2-NEXT:       Dead Store Elimination
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Memory SSA
; GCN-O2-NEXT:       Natural Loop Information
; GCN-O2-NEXT:       Canonicalize natural loops
; GCN-O2-NEXT:       LCSSA Verifier
; GCN-O2-NEXT:       Loop-Closed SSA Form Pass
; GCN-O2-NEXT:       Scalar Evolution Analysis
; GCN-O2-NEXT:       Loop Pass Manager
; GCN-O2-NEXT:         Loop Invariant Code Motion
; GCN-O2-NEXT:       Post-Dominator Tree Construction
; GCN-O2-NEXT:       Aggressive Dead Code Elimination
; GCN-O2-NEXT:       Simplify the CFG
; GCN-O2-NEXT:       Dominator Tree Construction
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Natural Loop Information
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Combine redundant instructions
; GCN-O2-NEXT:     A No-Op Barrier Pass
; GCN-O2-NEXT:     Eliminate Available Externally Globals
; GCN-O2-NEXT:     CallGraph Construction
; GCN-O2-NEXT:     Deduce function attributes in RPO
; GCN-O2-NEXT:     Global Variable Optimizer
; GCN-O2-NEXT:       FunctionPass Manager
; GCN-O2-NEXT:         Dominator Tree Construction
; GCN-O2-NEXT:         Natural Loop Information
; GCN-O2-NEXT:         Post-Dominator Tree Construction
; GCN-O2-NEXT:         Branch Probability Analysis
; GCN-O2-NEXT:         Block Frequency Analysis
; GCN-O2-NEXT:     Dead Global Elimination
; GCN-O2-NEXT:     CallGraph Construction
; GCN-O2-NEXT:     Globals Alias Analysis
; GCN-O2-NEXT:     FunctionPass Manager
; GCN-O2-NEXT:       Dominator Tree Construction
; GCN-O2-NEXT:       Float to int
; GCN-O2-NEXT:       Lower constant intrinsics
; GCN-O2-NEXT:       Dominator Tree Construction
; GCN-O2-NEXT:       Natural Loop Information
; GCN-O2-NEXT:       Canonicalize natural loops
; GCN-O2-NEXT:       LCSSA Verifier
; GCN-O2-NEXT:       Loop-Closed SSA Form Pass
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Scalar Evolution Analysis
; GCN-O2-NEXT:       Loop Pass Manager
; GCN-O2-NEXT:         Rotate Loops
; GCN-O2-NEXT:       Loop Access Analysis
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Loop Distribution
; GCN-O2-NEXT:       Post-Dominator Tree Construction
; GCN-O2-NEXT:       Branch Probability Analysis
; GCN-O2-NEXT:       Block Frequency Analysis
; GCN-O2-NEXT:       Scalar Evolution Analysis
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Loop Access Analysis
; GCN-O2-NEXT:       Demanded bits analysis
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Inject TLI Mappings
; GCN-O2-NEXT:       Loop Vectorization
; GCN-O2-NEXT:       Optimize scalar/vector ops
; GCN-O2-NEXT:       Early CSE
; GCN-O2-NEXT:       Canonicalize natural loops
; GCN-O2-NEXT:       Scalar Evolution Analysis
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Loop Access Analysis
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Loop Load Elimination
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Combine redundant instructions
; GCN-O2-NEXT:       Simplify the CFG
; GCN-O2-NEXT:       Dominator Tree Construction
; GCN-O2-NEXT:       Natural Loop Information
; GCN-O2-NEXT:       Scalar Evolution Analysis
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Demanded bits analysis
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Inject TLI Mappings
; GCN-O2-NEXT:       SLP Vectorizer
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Combine redundant instructions
; GCN-O2-NEXT:       Canonicalize natural loops
; GCN-O2-NEXT:       LCSSA Verifier
; GCN-O2-NEXT:       Loop-Closed SSA Form Pass
; GCN-O2-NEXT:       Scalar Evolution Analysis
; GCN-O2-NEXT:       Loop Pass Manager
; GCN-O2-NEXT:         Unroll loops
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Combine redundant instructions
; GCN-O2-NEXT:       Memory SSA
; GCN-O2-NEXT:       Canonicalize natural loops
; GCN-O2-NEXT:       LCSSA Verifier
; GCN-O2-NEXT:       Loop-Closed SSA Form Pass
; GCN-O2-NEXT:       Scalar Evolution Analysis
; GCN-O2-NEXT:       Loop Pass Manager
; GCN-O2-NEXT:         Loop Invariant Code Motion
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Warn about non-applied transformations
; GCN-O2-NEXT:       Alignment from assumptions
; GCN-O2-NEXT:     Strip Unused Function Prototypes
; GCN-O2-NEXT:     Dead Global Elimination
; GCN-O2-NEXT:     Merge Duplicate Global Constants
; GCN-O2-NEXT:     FunctionPass Manager
; GCN-O2-NEXT:       Dominator Tree Construction
; GCN-O2-NEXT:       Natural Loop Information
; GCN-O2-NEXT:       Post-Dominator Tree Construction
; GCN-O2-NEXT:       Branch Probability Analysis
; GCN-O2-NEXT:       Block Frequency Analysis
; GCN-O2-NEXT:       Canonicalize natural loops
; GCN-O2-NEXT:       LCSSA Verifier
; GCN-O2-NEXT:       Loop-Closed SSA Form Pass
; GCN-O2-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:       Function Alias Analysis Results
; GCN-O2-NEXT:       Scalar Evolution Analysis
; GCN-O2-NEXT:       Block Frequency Analysis
; GCN-O2-NEXT:       Loop Pass Manager
; GCN-O2-NEXT:         Loop Sink
; GCN-O2-NEXT:       Lazy Branch Probability Analysis
; GCN-O2-NEXT:       Lazy Block Frequency Analysis
; GCN-O2-NEXT:       Optimization Remark Emitter
; GCN-O2-NEXT:       Remove redundant instructions
; GCN-O2-NEXT:       Hoist/decompose integer division and remainder
; GCN-O2-NEXT:       Simplify the CFG

; GCN-O2-NEXT: Pass Arguments:
; GCN-O2-NEXT:   FunctionPass Manager
; GCN-O2-NEXT:     Dominator Tree Construction

; GCN-O2-NEXT: Pass Arguments:
; GCN-O2-NEXT:   FunctionPass Manager
; GCN-O2-NEXT:     Dominator Tree Construction

; GCN-O2-NEXT: Pass Arguments:
; GCN-O2-NEXT: Target Library Information
; GCN-O2-NEXT:   FunctionPass Manager
; GCN-O2-NEXT:     Dominator Tree Construction
; GCN-O2-NEXT:     Natural Loop Information
; GCN-O2-NEXT:     Post-Dominator Tree Construction
; GCN-O2-NEXT:     Branch Probability Analysis
; GCN-O2-NEXT:     Block Frequency Analysis

; GCN-O2-NEXT: Pass Arguments:
; GCN-O2-NEXT: Target Library Information
; GCN-O2-NEXT:   FunctionPass Manager
; GCN-O2-NEXT:     Dominator Tree Construction
; GCN-O2-NEXT:     Natural Loop Information
; GCN-O2-NEXT:     Post-Dominator Tree Construction
; GCN-O2-NEXT:     Branch Probability Analysis
; GCN-O2-NEXT:     Block Frequency Analysis


; GCN-O3:      Pass Arguments:
; GCN-O3-NEXT: Target Transform Information
; GCN-O3-NEXT: AMDGPU Address space based Alias Analysis
; GCN-O3-NEXT: External Alias Analysis
; GCN-O3-NEXT: Assumption Cache Tracker
; GCN-O3-NEXT: Target Library Information
; GCN-O3-NEXT: Type-Based Alias Analysis
; GCN-O3-NEXT: Scoped NoAlias Alias Analysis
; GCN-O3-NEXT:   FunctionPass Manager
; GCN-O3-NEXT:     Early propagate attributes from kernels to functions
; GCN-O3-NEXT:     Replace builtin math calls with that native versions.
; GCN-O3-NEXT:     Dominator Tree Construction
; GCN-O3-NEXT:     Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:     Function Alias Analysis Results
; GCN-O3-NEXT:     Simplify well-known AMD library calls
; GCN-O3-NEXT:     Instrument function entry/exit with calls to e.g. mcount() (pre inlining)
; GCN-O3-NEXT:     Simplify the CFG
; GCN-O3-NEXT:     Dominator Tree Construction
; GCN-O3-NEXT:     SROA
; GCN-O3-NEXT:     Early CSE
; GCN-O3-NEXT:     Lower 'expect' Intrinsics

; GCN-O3-NEXT: Pass Arguments:
; GCN-O3-NEXT: Target Library Information
; GCN-O3-NEXT: Target Transform Information
; GCN-O3-NEXT: Target Pass Configuration
; GCN-O3-NEXT: Type-Based Alias Analysis
; GCN-O3-NEXT: Scoped NoAlias Alias Analysis
; GCN-O3-NEXT: AMDGPU Address space based Alias Analysis
; GCN-O3-NEXT: External Alias Analysis
; GCN-O3-NEXT: Assumption Cache Tracker
; GCN-O3-NEXT: Profile summary info
; GCN-O3-NEXT:   ModulePass Manager
; GCN-O3-NEXT:     Force set function attributes
; GCN-O3-NEXT:     Infer set function attributes
; GCN-O3-NEXT:     Unify multiple OpenCL metadata due to linking
; GCN-O3-NEXT:     AMDGPU Printf lowering
; GCN-O3-NEXT:       FunctionPass Manager
; GCN-O3-NEXT:         Dominator Tree Construction
; GCN-O3-NEXT:     Late propagate attributes from kernels to functions
; GCN-O3-NEXT:     FunctionPass Manager
; GCN-O3-NEXT:       Dominator Tree Construction
; GCN-O3-NEXT:       Call-site splitting
; GCN-O3-NEXT:     Interprocedural Sparse Conditional Constant Propagation
; GCN-O3-NEXT:       FunctionPass Manager
; GCN-O3-NEXT:         Dominator Tree Construction
; GCN-O3-NEXT:     Called Value Propagation
; GCN-O3-NEXT:     Global Variable Optimizer
; GCN-O3-NEXT:       FunctionPass Manager
; GCN-O3-NEXT:         Dominator Tree Construction
; GCN-O3-NEXT:         Natural Loop Information
; GCN-O3-NEXT:         Post-Dominator Tree Construction
; GCN-O3-NEXT:         Branch Probability Analysis
; GCN-O3-NEXT:         Block Frequency Analysis
; GCN-O3-NEXT:     FunctionPass Manager
; GCN-O3-NEXT:       Dominator Tree Construction
; GCN-O3-NEXT:       Promote Memory to Register
; GCN-O3-NEXT:     Dead Argument Elimination
; GCN-O3-NEXT:     FunctionPass Manager
; GCN-O3-NEXT:       Dominator Tree Construction
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Natural Loop Information
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Combine redundant instructions
; GCN-O3-NEXT:       Simplify the CFG
; GCN-O3-NEXT:     CallGraph Construction
; GCN-O3-NEXT:     Globals Alias Analysis
; GCN-O3-NEXT:     Call Graph SCC Pass Manager
; GCN-O3-NEXT:       Remove unused exception handling info
; GCN-O3-NEXT:       AMDGPU Function Integration/Inlining
; GCN-O3-NEXT:       OpenMP specific optimizations
; GCN-O3-NEXT:       Deduce function attributes
; GCN-O3-NEXT:       Promote 'by reference' arguments to scalars
; GCN-O3-NEXT:       FunctionPass Manager
; GCN-O3-NEXT:         Infer address spaces
; GCN-O3-NEXT:     AMDGPU Kernel Attributes
; GCN-O3-NEXT:     FunctionPass Manager
; GCN-O3-NEXT:       AMDGPU Promote Alloca to vector
; GCN-O3-NEXT:       Dominator Tree Construction
; GCN-O3-NEXT:       SROA
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Memory SSA
; GCN-O3-NEXT:       Early CSE w/ MemorySSA
; GCN-O3-NEXT:       Speculatively execute instructions if target has divergent branches
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Lazy Value Information Analysis
; GCN-O3-NEXT:       Jump Threading
; GCN-O3-NEXT:       Value Propagation
; GCN-O3-NEXT:       Simplify the CFG
; GCN-O3-NEXT:       Dominator Tree Construction
; GCN-O3-NEXT:       Combine pattern based expressions
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Natural Loop Information
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Combine redundant instructions
; GCN-O3-NEXT:       Conditionally eliminate dead library calls
; GCN-O3-NEXT:       Natural Loop Information
; GCN-O3-NEXT:       Post-Dominator Tree Construction
; GCN-O3-NEXT:       Branch Probability Analysis
; GCN-O3-NEXT:       Block Frequency Analysis
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       PGOMemOPSize
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Natural Loop Information
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Tail Call Elimination
; GCN-O3-NEXT:       Simplify the CFG
; GCN-O3-NEXT:       Reassociate expressions
; GCN-O3-NEXT:       Dominator Tree Construction
; GCN-O3-NEXT:       Natural Loop Information
; GCN-O3-NEXT:       Canonicalize natural loops
; GCN-O3-NEXT:       LCSSA Verifier
; GCN-O3-NEXT:       Loop-Closed SSA Form Pass
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Scalar Evolution Analysis
; GCN-O3-NEXT:       Loop Pass Manager
; GCN-O3-NEXT:         Rotate Loops
; GCN-O3-NEXT:       Memory SSA
; GCN-O3-NEXT:       Loop Pass Manager
; GCN-O3-NEXT:         Loop Invariant Code Motion
; GCN-O3-NEXT:       Post-Dominator Tree Construction
; GCN-O3-NEXT:       Legacy Divergence Analysis
; GCN-O3-NEXT:       Loop Pass Manager
; GCN-O3-NEXT:         Unswitch loops
; GCN-O3-NEXT:       Simplify the CFG
; GCN-O3-NEXT:       Dominator Tree Construction
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Natural Loop Information
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Combine redundant instructions
; GCN-O3-NEXT:       Canonicalize natural loops
; GCN-O3-NEXT:       LCSSA Verifier
; GCN-O3-NEXT:       Loop-Closed SSA Form Pass
; GCN-O3-NEXT:       Scalar Evolution Analysis
; GCN-O3-NEXT:       Loop Pass Manager
; GCN-O3-NEXT:         Induction Variable Simplification
; GCN-O3-NEXT:         Recognize loop idioms
; GCN-O3-NEXT:         Delete dead loops
; GCN-O3-NEXT:         Unroll loops
; GCN-O3-NEXT:       MergedLoadStoreMotion
; GCN-O3-NEXT:       Phi Values Analysis
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Memory Dependence Analysis
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Global Value Numbering
; GCN-O3-NEXT:       Phi Values Analysis
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Memory Dependence Analysis
; GCN-O3-NEXT:       MemCpy Optimization
; GCN-O3-NEXT:       Sparse Conditional Constant Propagation
; GCN-O3-NEXT:       Demanded bits analysis
; GCN-O3-NEXT:       Bit-Tracking Dead Code Elimination
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Combine redundant instructions
; GCN-O3-NEXT:       Lazy Value Information Analysis
; GCN-O3-NEXT:       Jump Threading
; GCN-O3-NEXT:       Value Propagation
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Phi Values Analysis
; GCN-O3-NEXT:       Memory Dependence Analysis
; GCN-O3-NEXT:       Dead Store Elimination
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Memory SSA
; GCN-O3-NEXT:       Natural Loop Information
; GCN-O3-NEXT:       Canonicalize natural loops
; GCN-O3-NEXT:       LCSSA Verifier
; GCN-O3-NEXT:       Loop-Closed SSA Form Pass
; GCN-O3-NEXT:       Scalar Evolution Analysis
; GCN-O3-NEXT:       Loop Pass Manager
; GCN-O3-NEXT:         Loop Invariant Code Motion
; GCN-O3-NEXT:       Post-Dominator Tree Construction
; GCN-O3-NEXT:       Aggressive Dead Code Elimination
; GCN-O3-NEXT:       Simplify the CFG
; GCN-O3-NEXT:       Dominator Tree Construction
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Natural Loop Information
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Combine redundant instructions
; GCN-O3-NEXT:     A No-Op Barrier Pass
; GCN-O3-NEXT:     Eliminate Available Externally Globals
; GCN-O3-NEXT:     CallGraph Construction
; GCN-O3-NEXT:     Deduce function attributes in RPO
; GCN-O3-NEXT:     Global Variable Optimizer
; GCN-O3-NEXT:       FunctionPass Manager
; GCN-O3-NEXT:         Dominator Tree Construction
; GCN-O3-NEXT:         Natural Loop Information
; GCN-O3-NEXT:         Post-Dominator Tree Construction
; GCN-O3-NEXT:         Branch Probability Analysis
; GCN-O3-NEXT:         Block Frequency Analysis
; GCN-O3-NEXT:     Dead Global Elimination
; GCN-O3-NEXT:     CallGraph Construction
; GCN-O3-NEXT:     Globals Alias Analysis
; GCN-O3-NEXT:     FunctionPass Manager
; GCN-O3-NEXT:       Dominator Tree Construction
; GCN-O3-NEXT:       Float to int
; GCN-O3-NEXT:       Lower constant intrinsics
; GCN-O3-NEXT:       Dominator Tree Construction
; GCN-O3-NEXT:       Natural Loop Information
; GCN-O3-NEXT:       Canonicalize natural loops
; GCN-O3-NEXT:       LCSSA Verifier
; GCN-O3-NEXT:       Loop-Closed SSA Form Pass
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Scalar Evolution Analysis
; GCN-O3-NEXT:       Loop Pass Manager
; GCN-O3-NEXT:         Rotate Loops
; GCN-O3-NEXT:       Loop Access Analysis
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Loop Distribution
; GCN-O3-NEXT:       Post-Dominator Tree Construction
; GCN-O3-NEXT:       Branch Probability Analysis
; GCN-O3-NEXT:       Block Frequency Analysis
; GCN-O3-NEXT:       Scalar Evolution Analysis
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Loop Access Analysis
; GCN-O3-NEXT:       Demanded bits analysis
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Inject TLI Mappings
; GCN-O3-NEXT:       Loop Vectorization
; GCN-O3-NEXT:       Optimize scalar/vector ops
; GCN-O3-NEXT:       Early CSE
; GCN-O3-NEXT:       Canonicalize natural loops
; GCN-O3-NEXT:       Scalar Evolution Analysis
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Loop Access Analysis
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Loop Load Elimination
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Combine redundant instructions
; GCN-O3-NEXT:       Simplify the CFG
; GCN-O3-NEXT:       Dominator Tree Construction
; GCN-O3-NEXT:       Natural Loop Information
; GCN-O3-NEXT:       Scalar Evolution Analysis
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Demanded bits analysis
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Inject TLI Mappings
; GCN-O3-NEXT:       SLP Vectorizer
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Combine redundant instructions
; GCN-O3-NEXT:       Canonicalize natural loops
; GCN-O3-NEXT:       LCSSA Verifier
; GCN-O3-NEXT:       Loop-Closed SSA Form Pass
; GCN-O3-NEXT:       Scalar Evolution Analysis
; GCN-O3-NEXT:       Loop Pass Manager
; GCN-O3-NEXT:         Unroll loops
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Combine redundant instructions
; GCN-O3-NEXT:       Memory SSA
; GCN-O3-NEXT:       Canonicalize natural loops
; GCN-O3-NEXT:       LCSSA Verifier
; GCN-O3-NEXT:       Loop-Closed SSA Form Pass
; GCN-O3-NEXT:       Scalar Evolution Analysis
; GCN-O3-NEXT:       Loop Pass Manager
; GCN-O3-NEXT:         Loop Invariant Code Motion
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Warn about non-applied transformations
; GCN-O3-NEXT:       Alignment from assumptions
; GCN-O3-NEXT:     Strip Unused Function Prototypes
; GCN-O3-NEXT:     Dead Global Elimination
; GCN-O3-NEXT:     Merge Duplicate Global Constants
; GCN-O3-NEXT:     FunctionPass Manager
; GCN-O3-NEXT:       Dominator Tree Construction
; GCN-O3-NEXT:       Natural Loop Information
; GCN-O3-NEXT:       Post-Dominator Tree Construction
; GCN-O3-NEXT:       Branch Probability Analysis
; GCN-O3-NEXT:       Block Frequency Analysis
; GCN-O3-NEXT:       Canonicalize natural loops
; GCN-O3-NEXT:       LCSSA Verifier
; GCN-O3-NEXT:       Loop-Closed SSA Form Pass
; GCN-O3-NEXT:       Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:       Function Alias Analysis Results
; GCN-O3-NEXT:       Scalar Evolution Analysis
; GCN-O3-NEXT:       Block Frequency Analysis
; GCN-O3-NEXT:       Loop Pass Manager
; GCN-O3-NEXT:         Loop Sink
; GCN-O3-NEXT:       Lazy Branch Probability Analysis
; GCN-O3-NEXT:       Lazy Block Frequency Analysis
; GCN-O3-NEXT:       Optimization Remark Emitter
; GCN-O3-NEXT:       Remove redundant instructions
; GCN-O3-NEXT:       Hoist/decompose integer division and remainder
; GCN-O3-NEXT:       Simplify the CFG

; GCN-O3-NEXT: Pass Arguments:
; GCN-O3-NEXT:   FunctionPass Manager
; GCN-O3-NEXT:     Dominator Tree Construction

; GCN-O3-NEXT: Pass Arguments:
; GCN-O3-NEXT:   FunctionPass Manager
; GCN-O3-NEXT:     Dominator Tree Construction

; GCN-O3-NEXT: Pass Arguments:
; GCN-O3-NEXT: Target Library Information
; GCN-O3-NEXT:   FunctionPass Manager
; GCN-O3-NEXT:     Dominator Tree Construction
; GCN-O3-NEXT:     Natural Loop Information
; GCN-O3-NEXT:     Post-Dominator Tree Construction
; GCN-O3-NEXT:     Branch Probability Analysis
; GCN-O3-NEXT:     Block Frequency Analysis

; GCN-O3-NEXT: Pass Arguments:
; GCN-O3-NEXT: Target Library Information
; GCN-O3-NEXT:   FunctionPass Manager
; GCN-O3-NEXT:     Dominator Tree Construction
; GCN-O3-NEXT:     Natural Loop Information
; GCN-O3-NEXT:     Post-Dominator Tree Construction
; GCN-O3-NEXT:     Branch Probability Analysis
; GCN-O3-NEXT:     Block Frequency Analysis

define void @empty() {
  ret void
}
