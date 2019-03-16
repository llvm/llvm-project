; RUN: llc -mtriple=arm -O3 -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck %s

; REQUIRES: asserts

; CHECK:  ModulePass Manager
; CHECK:    Pre-ISel Intrinsic Lowering
; CHECK:    FunctionPass Manager
; CHECK:      Expand Atomic instructions
; CHECK:      Simplify the CFG
; CHECK:      Dominator Tree Construction
; CHECK:      Basic Alias Analysis (stateless AA impl)
; CHECK:      Module Verifier
; CHECK:      Natural Loop Information
; CHECK:      Canonicalize natural loops
; CHECK:      Scalar Evolution Analysis
; CHECK:      Loop Pass Manager
; CHECK:        Induction Variable Users
; CHECK:        Loop Strength Reduction
; CHECK:      Basic Alias Analysis (stateless AA impl)
; CHECK:      Function Alias Analysis Results
; CHECK:      Merge contiguous icmps into a memcmp
; CHECK:      Expand memcmp() to load/stores
; CHECK:      Lower Garbage Collection Instructions
; CHECK:      Shadow Stack GC Lowering
; CHECK:      Remove unreachable blocks from the CFG
; CHECK:      Dominator Tree Construction
; CHECK:      Natural Loop Information
; CHECK:      Branch Probability Analysis
; CHECK:      Block Frequency Analysis
; CHECK:      Constant Hoisting
; CHECK:      Partially inline calls to library functions
; CHECK:      Instrument function entry/exit with calls to e.g. mcount() (post inlining)
; CHECK:      Scalarize Masked Memory Intrinsics
; CHECK:      Expand reduction intrinsics
; CHECK:      Dominator Tree Construction
; CHECK:      Natural Loop Information
; CHECK:      Scalar Evolution Analysis
; CHECK:      Basic Alias Analysis (stateless AA impl)
; CHECK:      Function Alias Analysis Results
; CHECK:      Loop Pass Manager
; CHECK:        Transform loops to use DSP intrinsics
; CHECK:      Interleaved Access Pass
; CHECK:      ARM IR optimizations
; CHECK:      Dominator Tree Construction
; CHECK:      Natural Loop Information
; CHECK:      CodeGen Prepare
; CHECK:    Rewrite Symbols
; CHECK:    FunctionPass Manager
; CHECK:      Dominator Tree Construction
; CHECK:      Exception handling preparation
; CHECK:      Merge internal globals
; CHECK:      Safe Stack instrumentation pass
; CHECK:      Insert stack protectors
; CHECK:      Module Verifier
; CHECK:      Dominator Tree Construction
; CHECK:      Basic Alias Analysis (stateless AA impl)
; CHECK:      Function Alias Analysis Results
; CHECK:      Natural Loop Information
; CHECK:      Branch Probability Analysis
; CHECK:      ARM Instruction Selection
; CHECK:      Expand ISel Pseudo-instructions
; CHECK:      Early Tail Duplication
; CHECK:      Optimize machine instruction PHIs
; CHECK:      Slot index numbering
; CHECK:      Merge disjoint stack slots
; CHECK:      Local Stack Slot Allocation
; CHECK:      Remove dead machine instructions
; CHECK:      MachineDominator Tree Construction
; CHECK:      Machine Natural Loop Construction
; CHECK:      Early Machine Loop Invariant Code Motion
; CHECK:      Machine Common Subexpression Elimination
; CHECK:      MachinePostDominator Tree Construction
; CHECK:      Machine Block Frequency Analysis
; CHECK:      Machine code sinking
; CHECK:      Peephole Optimizations
; CHECK:      Remove dead machine instructions
; CHECK:      ARM MLA / MLS expansion pass
; CHECK:      ARM pre- register allocation load / store optimization pass
; CHECK:      ARM A15 S->D optimizer
; CHECK:      Detect Dead Lanes
; CHECK:      Process Implicit Definitions
; CHECK:      Remove unreachable machine basic blocks
; CHECK:      Live Variable Analysis
; CHECK:      MachineDominator Tree Construction
; CHECK:      Machine Natural Loop Construction
; CHECK:      Eliminate PHI nodes for register allocation
; CHECK:      Two-Address instruction pass
; CHECK:      Slot index numbering
; CHECK:      Live Interval Analysis
; CHECK:      Simple Register Coalescing
; CHECK:      Rename Disconnected Subregister Components
; CHECK:      Machine Instruction Scheduler
; CHECK:      Machine Block Frequency Analysis
; CHECK:      Debug Variable Analysis
; CHECK:      Live Stack Slot Analysis
; CHECK:      Virtual Register Map
; CHECK:      Live Register Matrix
; CHECK:      Bundle Machine CFG Edges
; CHECK:      Spill Code Placement Analysis
; CHECK:      Lazy Machine Block Frequency Analysis
; CHECK:      Machine Optimization Remark Emitter
; CHECK:      Greedy Register Allocator
; CHECK:      Virtual Register Rewriter
; CHECK:      Stack Slot Coloring
; CHECK:      Machine Copy Propagation Pass
; CHECK:      Machine Loop Invariant Code Motion
; CHECK:      PostRA Machine Sink
; CHECK:      Machine Block Frequency Analysis
; CHECK:      MachinePostDominator Tree Construction
; CHECK:      Lazy Machine Block Frequency Analysis
; CHECK:      Machine Optimization Remark Emitter
; CHECK:      Shrink Wrapping analysis
; CHECK:      Prologue/Epilogue Insertion & Frame Finalization
; CHECK:      Control Flow Optimizer
; CHECK:      Tail Duplication
; CHECK:      Machine Copy Propagation Pass
; CHECK:      Post-RA pseudo instruction expansion pass
; CHECK:      ARM load / store optimization pass
; CHECK:      ReachingDefAnalysis
; CHECK:      ARM Execution Domain Fix
; CHECK:      BreakFalseDeps
; CHECK:      ARM pseudo instruction expansion pass
; CHECK:      Thumb2 instruction size reduce pass
; CHECK:      MachineDominator Tree Construction
; CHECK:      Machine Natural Loop Construction
; CHECK:      Machine Block Frequency Analysis
; CHECK:      If Converter
; CHECK:      Thumb IT blocks insertion pass
; CHECK:      MachineDominator Tree Construction
; CHECK:      Machine Natural Loop Construction
; CHECK:      Post RA top-down list latency scheduler
; CHECK:      Analyze Machine Code For Garbage Collection
; CHECK:      Machine Block Frequency Analysis
; CHECK:      MachinePostDominator Tree Construction
; CHECK:      Branch Probability Basic Block Placement
; CHECK:      Thumb2 instruction size reduce pass
; CHECK:      Unpack machine instruction bundles
; CHECK:      optimise barriers pass
; CHECK:      ARM constant island placement and branch shortening pass
; CHECK:      Contiguously Lay Out Funclets
; CHECK:      StackMap Liveness Analysis
; CHECK:      Live DEBUG_VALUE analysis
; CHECK:      Insert fentry calls
; CHECK:      Insert XRay ops
; CHECK:      Implement the 'patchable-function' attribute
; CHECK:      Lazy Machine Block Frequency Analysis
; CHECK:      Machine Optimization Remark Emitter
; CHECK:      ARM Assembly Printer
; CHECK:      Free MachineFunction
