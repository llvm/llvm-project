; RUN: llc -O2 -mtriple=wasm32-unknown-unknown -disable-verify -verify-machineinstrs=0 -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck -match-full-lines -strict-whitespace -check-prefix=O2 %s
; REQUIRES: asserts

; O2:Target Library Information
; O2-NEXT:Runtime Library Function Analysis
; O2-NEXT:Target Pass Configuration
; O2-NEXT:Machine Module Information
; O2-NEXT:Target Transform Information
; O2-NEXT:Assumption Cache Tracker
; O2-NEXT:Library Function Lowering Analysis
; O2-NEXT:Type-Based Alias Analysis
; O2-NEXT:Scoped NoAlias Alias Analysis
; O2-NEXT:Create Garbage Collector Module Metadata
; O2-NEXT:Profile summary info
; O2-NEXT:Machine Branch Probability Analysis
; O2-NEXT:  ModulePass Manager
; O2-NEXT:    FunctionPass Manager
; O2-NEXT:      Dominator Tree Construction
; O2-NEXT:      Basic Alias Analysis (stateless AA impl)
; O2-NEXT:      Function Alias Analysis Results
; O2-NEXT:      ObjC ARC contraction
; O2-NEXT:    Pre-ISel Intrinsic Lowering
; O2-NEXT:    FunctionPass Manager
; O2-NEXT:      Expand IR instructions
; O2-NEXT:    Add prototypes to prototypes-less functions
; O2-NEXT:    Lower @llvm.global_dtors via `__cxa_atexit`
; O2-NEXT:    WebAssembly Fix Function Bitcasts
; O2-NEXT:    FunctionPass Manager
; O2-NEXT:      Dominator Tree Construction
; O2-NEXT:      WebAssembly Optimize Returned
; O2-NEXT:    WebAssembly Lower Emscripten Exceptions
; O2-NEXT:      FunctionPass Manager
; O2-NEXT:        Dominator Tree Construction
; O2-NEXT:    FunctionPass Manager
; O2-NEXT:      Dominator Tree Construction
; O2-NEXT:      Cycle Info Analysis
; O2-NEXT:      Lazy Branch Probability Analysis
; O2-NEXT:      Lazy Block Frequency Analysis
; O2-NEXT:      Expand indirectbr instructions
; O2-NEXT:      WebAssembly convert reduce to any_true/all_true
; O2-NEXT:      Dominator Tree Construction
; O2-NEXT:      Basic Alias Analysis (stateless AA impl)
; O2-NEXT:      Natural Loop Information
; O2-NEXT:      Canonicalize natural loops
; O2-NEXT:      Scalar Evolution Analysis
; O2-NEXT:      Loop Pass Manager
; O2-NEXT:        Canonicalize Freeze Instructions in Loops
; O2-NEXT:        Induction Variable Users
; O2-NEXT:        Loop Strength Reduction
; O2-NEXT:      Lower Garbage Collection Instructions
; O2-NEXT:      Shadow Stack GC Lowering
; O2-NEXT:      Remove unreachable blocks from the CFG
; O2-NEXT:      Cycle Info Analysis
; O2-NEXT:      Lazy Branch Probability Analysis
; O2-NEXT:      Lazy Block Frequency Analysis
; O2-NEXT:      Constant Hoisting
; O2-NEXT:      Replace intrinsics with calls to vector library
; O2-NEXT:      Optimization Remark Emitter
; O2-NEXT:      Partially inline calls to library functions
; O2-NEXT:      Instrument function entry/exit with calls to e.g. mcount() (post inlining)
; O2-NEXT:      Scalarize Masked Memory Intrinsics
; O2-NEXT:      Expand reduction intrinsics
; O2-NEXT:      Natural Loop Information
; O2-NEXT:      Cycle Info Analysis
; O2-NEXT:      Post-Dominator Tree Construction
; O2-NEXT:      Branch Probability Analysis
; O2-NEXT:      Block Frequency Analysis
; O2-NEXT:      CodeGen Prepare
; O2-NEXT:      Windows exception handling preparation
; O2-NEXT:      WebAssembly Exception handling preparation
; O2-NEXT:      WebAssembly Reference Types Memory to Local
; O2-NEXT:    Unnamed pass: implement Pass::getPassName()
; O2-NEXT:    FunctionPass Manager
; O2-NEXT:      Expand Atomic instructions
; O2-NEXT:      Prepare inline asm insts
; O2-NEXT:      Safe Stack instrumentation pass
; O2-NEXT:      Insert stack protectors
; O2-NEXT:      Dominator Tree Construction
; O2-NEXT:      Basic Alias Analysis (stateless AA impl)
; O2-NEXT:      Function Alias Analysis Results
; O2-NEXT:      Cycle Info Analysis
; O2-NEXT:      Post-Dominator Tree Construction
; O2-NEXT:      Branch Probability Analysis
; O2-NEXT:      Assignment Tracking Analysis
; O2-NEXT:      Lazy Branch Probability Analysis
; O2-NEXT:      Lazy Block Frequency Analysis
; O2-NEXT:      WebAssembly Instruction Selection
; O2-NEXT:      WebAssembly Argument Move
; O2-NEXT:      WebAssembly Set p2align Operands
; O2-NEXT:      WebAssembly Fix br_table Defaults
; O2-NEXT:      WebAssembly Clean Code After Trap
; O2-NEXT:      Finalize ISel and expand pseudo-instructions
; O2-NEXT:      Lazy Machine Block Frequency Analysis
; O2-NEXT:      Early Tail Duplication
; O2-NEXT:      Optimize machine instruction PHIs
; O2-NEXT:      Slot index numbering
; O2-NEXT:      Merge disjoint stack slots
; O2-NEXT:      Local Stack Slot Allocation
; O2-NEXT:      Remove dead machine instructions
; O2-NEXT:      MachineDominator Tree Construction
; O2-NEXT:      Machine Natural Loop Construction
; O2-NEXT:      Lazy Machine Block Frequency Analysis
; O2-NEXT:      Machine Register Class Info Analysis
; O2-NEXT:      Early Machine Loop Invariant Code Motion
; O2-NEXT:      Machine Cycle Info Analysis
; O2-NEXT:      Machine Block Frequency Analysis
; O2-NEXT:      Machine Common Subexpression Elimination
; O2-NEXT:      MachinePostDominator Tree Construction
; O2-NEXT:      Machine code sinking
; O2-NEXT:      Peephole Optimizations
; O2-NEXT:      Remove dead machine instructions
; O2-NEXT:      Detect Dead Lanes
; O2-NEXT:      Init Undef Pass
; O2-NEXT:      Process Implicit Definitions
; O2-NEXT:      Remove unreachable machine basic blocks
; O2-NEXT:      MachineDominator Tree Construction
; O2-NEXT:      Slot index numbering
; O2-NEXT:      Live Interval Analysis
; O2-NEXT:      Eliminate PHI nodes for register allocation
; O2-NEXT:      Two-Address instruction pass
; O2-NEXT:      Register Coalescer
; O2-NEXT:      Rename Disconnected Subregister Components
; O2-NEXT:      Machine Instruction Scheduler
; O2-NEXT:      Remove Redundant DEBUG_VALUE analysis
; O2-NEXT:      Fixup Statepoint Caller Saved
; O2-NEXT:      Lazy Machine Block Frequency Analysis
; O2-NEXT:      Machine Optimization Remark Emitter
; O2-NEXT:      Prologue/Epilogue Insertion & Frame Finalization
; O2-NEXT:      Control Flow Optimizer
; O2-NEXT:      Lazy Machine Block Frequency Analysis
; O2-NEXT:      Tail Duplication
; O2-NEXT:      Post-RA pseudo instruction expansion pass
; O2-NEXT:      Insert fentry calls
; O2-NEXT:      Insert XRay ops
; O2-NEXT:      WebAssembly Nullify DBG_VALUE_LISTs
; O2-NEXT:      Remove unreachable machine basic blocks
; O2-NEXT:      WebAssembly Fix Irreducible Control Flow
; O2-NEXT:      WebAssembly Late Prepare Exception
; O2-NEXT:      WebAssembly Replace Physical Registers
; O2-NEXT:      MachineDominator Tree Construction
; O2-NEXT:      Slot index numbering
; O2-NEXT:      Live Interval Analysis
; O2-NEXT:      WebAssembly Optimize Live Intervals
; O2-NEXT:      WebAssembly Memory Intrinsic Results
; O2-NEXT:      WebAssembly Register Stackify
; O2-NEXT:      Machine Cycle Info Analysis
; O2-NEXT:      Machine Block Frequency Analysis
; O2-NEXT:      WebAssembly Register Coloring
; O2-NEXT:      Machine Natural Loop Construction
; O2-NEXT:      Machine Dominance Frontier Construction
; O2-NEXT:      WebAssembly Exception Information
; O2-NEXT:      WebAssembly CFG Sort
; O2-NEXT:      WebAssembly CFG Stackify
; O2-NEXT:      WebAssembly Explicit Locals
; O2-NEXT:      WebAssembly Lower br_unless
; O2-NEXT:      WebAssembly late peephole optimizer
; O2-NEXT:      WebAssembly Register Numbering
; O2-NEXT:      WebAssembly Debug Fixup
; O2-NEXT:    WebAssembly MC Lower Pre Pass
; O2-NEXT:    FunctionPass Manager
; O2-NEXT:      Live DEBUG_VALUE analysis
; O2-NEXT:      Machine Sanitizer Binary Metadata
; O2-NEXT:      Lazy Machine Block Frequency Analysis
; O2-NEXT:      Machine Optimization Remark Emitter
; O2-NEXT:      Stack Frame Layout Analysis
; O2-NEXT:      WebAssembly Assembly Printer
; O2-NEXT:      Free MachineFunction
; O2-NEXT:Pass Arguments:  -domtree
; O2-NEXT:  FunctionPass Manager
; O2-NEXT:    Dominator Tree Construction
