; UNSUPPORTED: expensive_checks
; RUN: llc -O0 -mtriple=amdgcn--amdhsa -amdgpu-wave-transform=1 -disable-verify -debug-pass=Structure < %s 2>&1 \
; RUN:   | FileCheck -match-full-lines -strict-whitespace -check-prefix=GCN-O0 %s
; RUN: llc -O3 -mtriple=amdgcn--amdhsa -amdgpu-wave-transform=1 -disable-verify -debug-pass=Structure < %s 2>&1 \
; RUN:   | FileCheck -match-full-lines -strict-whitespace -check-prefix=GCN-O3 %s

; REQUIRES: asserts

; GCN-O0:Target Library Information
; GCN-O0-NEXT:Target Pass Configuration
; GCN-O0-NEXT:Machine Module Information
; GCN-O0-NEXT:Target Transform Information
; GCN-O0-NEXT:Assumption Cache Tracker
; GCN-O0-NEXT:Profile summary info
; GCN-O0-NEXT:Argument Register Usage Information Storage
; GCN-O0-NEXT:Create Garbage Collector Module Metadata
; GCN-O0-NEXT:Register Usage Information Storage
; GCN-O0-NEXT:Machine Branch Probability Analysis
; GCN-O0-NEXT:  ModulePass Manager
; GCN-O0-NEXT:    Verify Heterogeneous Debug Preconditions
; GCN-O0-NEXT:    Pre-ISel Intrinsic Lowering
; GCN-O0-NEXT:    FunctionPass Manager
; GCN-O0-NEXT:      Expand large div/rem
; GCN-O0-NEXT:      Expand large fp convert
; GCN-O0-NEXT:    AMDGPU Remove Incompatible Functions
; GCN-O0-NEXT:    AMDGPU Printf lowering
; GCN-O0-NEXT:    Lower ctors and dtors for AMDGPU
; GCN-O0-NEXT:    AMDGPU Lower Kernel Calls
; GCN-O0-NEXT:    Expand variadic functions
; GCN-O0-NEXT:    AMDGPU Inline All Functions
; GCN-O0-NEXT:    Inliner for always_inline functions
; GCN-O0-NEXT:      FunctionPass Manager
; GCN-O0-NEXT:        Dominator Tree Construction
; GCN-O0-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O0-NEXT:        Function Alias Analysis Results
; GCN-O0-NEXT:    Lower OpenCL enqueued blocks
; GCN-O0-NEXT:    AMDGPU Software lowering of LDS
; GCN-O0-NEXT:    Lower uses of LDS variables from non-kernel functions
; GCN-O0-NEXT:    FunctionPass Manager
; GCN-O0-NEXT:      Expand Atomic instructions
; GCN-O0-NEXT:      Remove unreachable blocks from the CFG
; GCN-O0-NEXT:      Instrument function entry/exit with calls to e.g. mcount() (post inlining)
; GCN-O0-NEXT:      Scalarize Masked Memory Intrinsics
; GCN-O0-NEXT:      Expand reduction intrinsics
; GCN-O0-NEXT:    CallGraph Construction
; GCN-O0-NEXT:    Call Graph SCC Pass Manager
; GCN-O0-NEXT:      AMDGPU Annotate Kernel Features
; GCN-O0-NEXT:      FunctionPass Manager
; GCN-O0-NEXT:        AMDGPU Lower Kernel Arguments
; GCN-O0-NEXT:    Lower buffer fat pointer operations to buffer resources
; GCN-O0-NEXT:    CallGraph Construction
; GCN-O0-NEXT:    Call Graph SCC Pass Manager
; GCN-O0-NEXT:      DummyCGSCCPass
; GCN-O0-NEXT:      FunctionPass Manager
; GCN-O0-NEXT:        Lazy Value Information Analysis
; GCN-O0-NEXT:        Lower SwitchInst's to branches
; GCN-O0-NEXT:        Lower invoke and unwind, for unwindless code generators
; GCN-O0-NEXT:        Remove unreachable blocks from the CFG
; GCN-O0-NEXT:        Post-Dominator Tree Construction
; GCN-O0-NEXT:        Dominator Tree Construction
; GCN-O0-NEXT:        Cycle Info Analysis
; GCN-O0-NEXT:        Uniformity Analysis
; GCN-O0-NEXT:        Unify divergent function exit nodes
; GCN-O0-NEXT:        Dominator Tree Construction
; GCN-O0-NEXT:        Cycle Info Analysis
; GCN-O0-NEXT:        Convert irreducible control-flow into natural loops
; GCN-O0-NEXT:        Natural Loop Information
; GCN-O0-NEXT:        Fixup each natural loop to have a single exit block
; GCN-O0-NEXT:        LCSSA Verifier
; GCN-O0-NEXT:        Loop-Closed SSA Form Pass
; GCN-O0-NEXT:      DummyCGSCCPass
; GCN-O0-NEXT:      FunctionPass Manager
; GCN-O0-NEXT:        Prepare callbr
; GCN-O0-NEXT:        Safe Stack instrumentation pass
; GCN-O0-NEXT:        Insert stack protectors
; GCN-O0-NEXT:        Dominator Tree Construction
; GCN-O0-NEXT:        Cycle Info Analysis
; GCN-O0-NEXT:        Uniformity Analysis
; GCN-O0-NEXT:        Assignment Tracking Analysis
; GCN-O0-NEXT:        AMDGPU DAG->DAG Pattern Instruction Selection
; GCN-O0-NEXT:        Finalize ISel and expand pseudo-instructions
; GCN-O0-NEXT:        Local Stack Slot Allocation
; GCN-O0-NEXT:        Register Usage Information Propagation
; GCN-O0-NEXT:        Machine Cycle Info Analysis
; GCN-O0-NEXT:        MachineDominator Tree Construction
; GCN-O0-NEXT:        Machine Uniformity Info Analysis
; GCN-O0-NEXT:        AMDGPU Pre Wave Transform
; GCN-O0-NEXT:        Eliminate PHI nodes for register allocation
; GCN-O0-NEXT:        Two-Address instruction pass
; GCN-O0-NEXT:        AMDGPU Pre-RA Long Branch Reg
; GCN-O0-NEXT:        Fast Register Allocator
; GCN-O0-NEXT:        Machine Cycle Info Analysis
; GCN-O0-NEXT:        GCN Control Flow Wave Transform
; GCN-O0-NEXT:        Fast Register Allocator
; GCN-O0-NEXT:        SI lower SGPR spill instructions
; GCN-O0-NEXT:        Slot index numbering
; GCN-O0-NEXT:        Live Interval Analysis
; GCN-O0-NEXT:        Virtual Register Map
; GCN-O0-NEXT:        Live Register Matrix
; GCN-O0-NEXT:        SI Pre-allocate WWM Registers
; GCN-O0-NEXT:        Fast Register Allocator
; GCN-O0-NEXT:        SI Lower WWM Copies
; GCN-O0-NEXT:        SI Fix VGPR copies
; GCN-O0-NEXT:        Remove Redundant DEBUG_VALUE analysis
; GCN-O0-NEXT:        Fixup Statepoint Caller Saved
; GCN-O0-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O0-NEXT:        Machine Optimization Remark Emitter
; GCN-O0-NEXT:        Prologue/Epilogue Insertion & Frame Finalization
; GCN-O0-NEXT:        Post-RA pseudo instruction expansion pass
; GCN-O0-NEXT:        SI post-RA bundler
; GCN-O0-NEXT:        Insert fentry calls
; GCN-O0-NEXT:        Insert XRay ops
; GCN-O0-NEXT:        SI Memory Legalizer
; GCN-O0-NEXT:        MachineDominator Tree Construction
; GCN-O0-NEXT:        Machine Natural Loop Construction
; GCN-O0-NEXT:        MachinePostDominator Tree Construction
; GCN-O0-NEXT:        SI insert wait instructions
; GCN-O0-NEXT:        Insert required mode register values
; GCN-O0-NEXT:        SI Final Branch Preparation
; GCN-O0-NEXT:        Post RA hazard recognizer
; GCN-O0-NEXT:        AMDGPU Insert waits for SGPR read hazards
; GCN-O0-NEXT:        Branch relaxation pass
; GCN-O0-NEXT:        AMDGPU Preload Kernel Arguments Prolog
; GCN-O0-NEXT:        Register Usage Information Collector Pass
; GCN-O0-NEXT:        Remove Loads Into Fake Uses
; GCN-O0-NEXT:        Live DEBUG_VALUE analysis
; GCN-O0-NEXT:        Machine Sanitizer Binary Metadata
; GCN-O0-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O0-NEXT:        Machine Optimization Remark Emitter
; GCN-O0-NEXT:        Stack Frame Layout Analysis
; GCN-O0-NEXT:        Function register usage analysis
; GCN-O0-NEXT:        AMDGPU Assembly Printer
; GCN-O0-NEXT:        Free MachineFunction

; GCN-O3:Target Library Information
; GCN-O3-NEXT:Target Pass Configuration
; GCN-O3-NEXT:Machine Module Information
; GCN-O3-NEXT:Target Transform Information
; GCN-O3-NEXT:Assumption Cache Tracker
; GCN-O3-NEXT:Profile summary info
; GCN-O3-NEXT:AMDGPU Address space based Alias Analysis
; GCN-O3-NEXT:External Alias Analysis
; GCN-O3-NEXT:Type-Based Alias Analysis
; GCN-O3-NEXT:Scoped NoAlias Alias Analysis
; GCN-O3-NEXT:Argument Register Usage Information Storage
; GCN-O3-NEXT:Create Garbage Collector Module Metadata
; GCN-O3-NEXT:Machine Branch Probability Analysis
; GCN-O3-NEXT:Register Usage Information Storage
; GCN-O3-NEXT:Default Regalloc Eviction Advisor
; GCN-O3-NEXT:Default Regalloc Priority Advisor
; GCN-O3-NEXT:  ModulePass Manager
; GCN-O3-NEXT:    Verify Heterogeneous Debug Preconditions
; GCN-O3-NEXT:    Pre-ISel Intrinsic Lowering
; GCN-O3-NEXT:    FunctionPass Manager
; GCN-O3-NEXT:      Expand large div/rem
; GCN-O3-NEXT:      Expand large fp convert
; GCN-O3-NEXT:    AMDGPU Remove Incompatible Functions
; GCN-O3-NEXT:    AMDGPU Printf lowering
; GCN-O3-NEXT:    Lower ctors and dtors for AMDGPU
; GCN-O3-NEXT:    AMDGPU Lower Kernel Calls
; GCN-O3-NEXT:    FunctionPass Manager
; GCN-O3-NEXT:      AMDGPU Image Intrinsic Optimizer
; GCN-O3-NEXT:    Expand variadic functions
; GCN-O3-NEXT:    AMDGPU Inline All Functions
; GCN-O3-NEXT:    Inliner for always_inline functions
; GCN-O3-NEXT:      FunctionPass Manager
; GCN-O3-NEXT:        Dominator Tree Construction
; GCN-O3-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:        Function Alias Analysis Results
; GCN-O3-NEXT:    Lower OpenCL enqueued blocks
; GCN-O3-NEXT:    AMDGPU Software lowering of LDS
; GCN-O3-NEXT:    Lower uses of LDS variables from non-kernel functions
; GCN-O3-NEXT:    FunctionPass Manager
; GCN-O3-NEXT:      Infer address spaces
; GCN-O3-NEXT:      Dominator Tree Construction
; GCN-O3-NEXT:      Cycle Info Analysis
; GCN-O3-NEXT:      Uniformity Analysis
; GCN-O3-NEXT:      AMDGPU atomic optimizations
; GCN-O3-NEXT:      Expand Atomic instructions
; GCN-O3-NEXT:      Dominator Tree Construction
; GCN-O3-NEXT:      Natural Loop Information
; GCN-O3-NEXT:      AMDGPU Promote Alloca
; GCN-O3-NEXT:      Split GEPs to a variadic base and a constant offset for better CSE
; GCN-O3-NEXT:      Scalar Evolution Analysis
; GCN-O3-NEXT:      Straight line strength reduction
; GCN-O3-NEXT:      Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:      Function Alias Analysis Results
; GCN-O3-NEXT:      Memory Dependence Analysis
; GCN-O3-NEXT:      Lazy Branch Probability Analysis
; GCN-O3-NEXT:      Lazy Block Frequency Analysis
; GCN-O3-NEXT:      Optimization Remark Emitter
; GCN-O3-NEXT:      Global Value Numbering
; GCN-O3-NEXT:      Scalar Evolution Analysis
; GCN-O3-NEXT:      Nary reassociation
; GCN-O3-NEXT:      Early CSE
; GCN-O3-NEXT:      Cycle Info Analysis
; GCN-O3-NEXT:      Uniformity Analysis
; GCN-O3-NEXT:      AMDGPU IR optimizations
; GCN-O3-NEXT:      Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:      Function Alias Analysis Results
; GCN-O3-NEXT:      Memory SSA
; GCN-O3-NEXT:      Canonicalize natural loops
; GCN-O3-NEXT:      LCSSA Verifier
; GCN-O3-NEXT:      Loop-Closed SSA Form Pass
; GCN-O3-NEXT:      Scalar Evolution Analysis
; GCN-O3-NEXT:      Lazy Branch Probability Analysis
; GCN-O3-NEXT:      Lazy Block Frequency Analysis
; GCN-O3-NEXT:      Loop Pass Manager
; GCN-O3-NEXT:        Loop Invariant Code Motion
; GCN-O3-NEXT:      Loop Pass Manager
; GCN-O3-NEXT:        Canonicalize Freeze Instructions in Loops
; GCN-O3-NEXT:        Induction Variable Users
; GCN-O3-NEXT:        Loop Strength Reduction
; GCN-O3-NEXT:      Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:      Function Alias Analysis Results
; GCN-O3-NEXT:      Merge contiguous icmps into a memcmp
; GCN-O3-NEXT:      Natural Loop Information
; GCN-O3-NEXT:      Lazy Branch Probability Analysis
; GCN-O3-NEXT:      Lazy Block Frequency Analysis
; GCN-O3-NEXT:      Expand memcmp() to load/stores
; GCN-O3-NEXT:      Remove unreachable blocks from the CFG
; GCN-O3-NEXT:      Natural Loop Information
; GCN-O3-NEXT:      Post-Dominator Tree Construction
; GCN-O3-NEXT:      Branch Probability Analysis
; GCN-O3-NEXT:      Block Frequency Analysis
; GCN-O3-NEXT:      Constant Hoisting
; GCN-O3-NEXT:      Replace intrinsics with calls to vector library
; GCN-O3-NEXT:      Lazy Branch Probability Analysis
; GCN-O3-NEXT:      Lazy Block Frequency Analysis
; GCN-O3-NEXT:      Optimization Remark Emitter
; GCN-O3-NEXT:      Partially inline calls to library functions
; GCN-O3-NEXT:      Instrument function entry/exit with calls to e.g. mcount() (post inlining)
; GCN-O3-NEXT:      Scalarize Masked Memory Intrinsics
; GCN-O3-NEXT:      Expand reduction intrinsics
; GCN-O3-NEXT:      Natural Loop Information
; GCN-O3-NEXT:      Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:      Function Alias Analysis Results
; GCN-O3-NEXT:      Memory Dependence Analysis
; GCN-O3-NEXT:      Lazy Branch Probability Analysis
; GCN-O3-NEXT:      Lazy Block Frequency Analysis
; GCN-O3-NEXT:      Optimization Remark Emitter
; GCN-O3-NEXT:      Global Value Numbering
; GCN-O3-NEXT:    CallGraph Construction
; GCN-O3-NEXT:    Call Graph SCC Pass Manager
; GCN-O3-NEXT:      AMDGPU Annotate Kernel Features
; GCN-O3-NEXT:      FunctionPass Manager
; GCN-O3-NEXT:        AMDGPU Lower Kernel Arguments
; GCN-O3-NEXT:    Lower buffer fat pointer operations to buffer resources
; GCN-O3-NEXT:    CallGraph Construction
; GCN-O3-NEXT:    Call Graph SCC Pass Manager
; GCN-O3-NEXT:      DummyCGSCCPass
; GCN-O3-NEXT:      FunctionPass Manager
; GCN-O3-NEXT:        Dominator Tree Construction
; GCN-O3-NEXT:        Natural Loop Information
; GCN-O3-NEXT:        CodeGen Prepare
; GCN-O3-NEXT:        Dominator Tree Construction
; GCN-O3-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:        Function Alias Analysis Results
; GCN-O3-NEXT:        Natural Loop Information
; GCN-O3-NEXT:        Scalar Evolution Analysis
; GCN-O3-NEXT:        GPU Load and Store Vectorizer
; GCN-O3-NEXT:        Lazy Value Information Analysis
; GCN-O3-NEXT:        Lower SwitchInst's to branches
; GCN-O3-NEXT:        Lower invoke and unwind, for unwindless code generators
; GCN-O3-NEXT:        Remove unreachable blocks from the CFG
; GCN-O3-NEXT:        Dominator Tree Construction
; GCN-O3-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:        Function Alias Analysis Results
; GCN-O3-NEXT:        Flatten the CFG
; GCN-O3-NEXT:        Dominator Tree Construction
; GCN-O3-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:        Function Alias Analysis Results
; GCN-O3-NEXT:        Natural Loop Information
; GCN-O3-NEXT:        Code sinking
; GCN-O3-NEXT:        Cycle Info Analysis
; GCN-O3-NEXT:        Uniformity Analysis
; GCN-O3-NEXT:        AMDGPU IR late optimizations
; GCN-O3-NEXT:        Post-Dominator Tree Construction
; GCN-O3-NEXT:        Unify divergent function exit nodes
; GCN-O3-NEXT:        Dominator Tree Construction
; GCN-O3-NEXT:        Cycle Info Analysis
; GCN-O3-NEXT:        Convert irreducible control-flow into natural loops
; GCN-O3-NEXT:        Natural Loop Information
; GCN-O3-NEXT:        Fixup each natural loop to have a single exit block
; GCN-O3-NEXT:        LCSSA Verifier
; GCN-O3-NEXT:        Loop-Closed SSA Form Pass
; GCN-O3-NEXT:      Analysis if a function is memory bound
; GCN-O3-NEXT:      DummyCGSCCPass
; GCN-O3-NEXT:      FunctionPass Manager
; GCN-O3-NEXT:        Dominator Tree Construction
; GCN-O3-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:        Function Alias Analysis Results
; GCN-O3-NEXT:        ObjC ARC contraction
; GCN-O3-NEXT:        Prepare callbr
; GCN-O3-NEXT:        Safe Stack instrumentation pass
; GCN-O3-NEXT:        Insert stack protectors
; GCN-O3-NEXT:        Cycle Info Analysis
; GCN-O3-NEXT:        Uniformity Analysis
; GCN-O3-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:        Function Alias Analysis Results
; GCN-O3-NEXT:        Natural Loop Information
; GCN-O3-NEXT:        Post-Dominator Tree Construction
; GCN-O3-NEXT:        Branch Probability Analysis
; GCN-O3-NEXT:        Assignment Tracking Analysis
; GCN-O3-NEXT:        Lazy Branch Probability Analysis
; GCN-O3-NEXT:        Lazy Block Frequency Analysis
; GCN-O3-NEXT:        AMDGPU DAG->DAG Pattern Instruction Selection
; GCN-O3-NEXT:        Finalize ISel and expand pseudo-instructions
; GCN-O3-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O3-NEXT:        Early Tail Duplication
; GCN-O3-NEXT:        Optimize machine instruction PHIs
; GCN-O3-NEXT:        Slot index numbering
; GCN-O3-NEXT:        Merge disjoint stack slots
; GCN-O3-NEXT:        Local Stack Slot Allocation
; GCN-O3-NEXT:        Remove dead machine instructions
; GCN-O3-NEXT:        MachineDominator Tree Construction
; GCN-O3-NEXT:        Machine Natural Loop Construction
; GCN-O3-NEXT:        Machine Block Frequency Analysis
; GCN-O3-NEXT:        Early Machine Loop Invariant Code Motion
; GCN-O3-NEXT:        MachineDominator Tree Construction
; GCN-O3-NEXT:        Machine Block Frequency Analysis
; GCN-O3-NEXT:        Machine Common Subexpression Elimination
; GCN-O3-NEXT:        MachinePostDominator Tree Construction
; GCN-O3-NEXT:        Machine Cycle Info Analysis
; GCN-O3-NEXT:        Machine code sinking
; GCN-O3-NEXT:        Peephole Optimizations
; GCN-O3-NEXT:        Remove dead machine instructions
; GCN-O3-NEXT:        SI Fold Operands
; GCN-O3-NEXT:        GCN DPP Combine
; GCN-O3-NEXT:        SI Load Store Optimizer
; GCN-O3-NEXT:        SI Peephole SDWA
; GCN-O3-NEXT:        Machine Block Frequency Analysis
; GCN-O3-NEXT:        MachineDominator Tree Construction
; GCN-O3-NEXT:        Early Machine Loop Invariant Code Motion
; GCN-O3-NEXT:        MachineDominator Tree Construction
; GCN-O3-NEXT:        Machine Block Frequency Analysis
; GCN-O3-NEXT:        Machine Common Subexpression Elimination
; GCN-O3-NEXT:        SI Fold Operands
; GCN-O3-NEXT:        Remove dead machine instructions
; GCN-O3-NEXT:        SI Shrink Instructions
; GCN-O3-NEXT:        Register Usage Information Propagation
; GCN-O3-NEXT:        Detect Dead Lanes
; GCN-O3-NEXT:        Remove dead machine instructions
; GCN-O3-NEXT:        Init Undef Pass
; GCN-O3-NEXT:        Process Implicit Definitions
; GCN-O3-NEXT:        Remove unreachable machine basic blocks
; GCN-O3-NEXT:        Live Variable Analysis
; GCN-O3-NEXT:        Machine Cycle Info Analysis
; GCN-O3-NEXT:        Machine Uniformity Info Analysis
; GCN-O3-NEXT:        AMDGPU Pre Wave Transform
; GCN-O3-NEXT:        Eliminate PHI nodes for register allocation
; GCN-O3-NEXT:        Two-Address instruction pass
; GCN-O3-NEXT:        Slot index numbering
; GCN-O3-NEXT:        Live Interval Analysis
; GCN-O3-NEXT:        Register Coalescer
; GCN-O3-NEXT:        Rename Disconnected Subregister Components
; GCN-O3-NEXT:        Rewrite Partial Register Uses
; GCN-O3-NEXT:        Machine Instruction Scheduler
; GCN-O3-NEXT:        AMDGPU Pre-RA optimizations
; GCN-O3-NEXT:        SI Form memory clauses
; GCN-O3-NEXT:        AMDGPU Pre-RA Long Branch Reg
; GCN-O3-NEXT:        Machine Block Frequency Analysis
; GCN-O3-NEXT:        Debug Variable Analysis
; GCN-O3-NEXT:        Live Stack Slot Analysis
; GCN-O3-NEXT:        Virtual Register Map
; GCN-O3-NEXT:        Live Register Matrix
; GCN-O3-NEXT:        Bundle Machine CFG Edges
; GCN-O3-NEXT:        Spill Code Placement Analysis
; GCN-O3-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O3-NEXT:        Machine Optimization Remark Emitter
; GCN-O3-NEXT:        Greedy Register Allocator
; GCN-O3-NEXT:        GCN NSA Reassign
; GCN-O3-NEXT:        Virtual Register Rewriter
; GCN-O3-NEXT:        Machine Cycle Info Analysis
; GCN-O3-NEXT:        GCN Control Flow Wave Transform
; GCN-O3-NEXT:        Slot index numbering
; GCN-O3-NEXT:        Live Interval Analysis
; GCN-O3-NEXT:        Machine Natural Loop Construction
; GCN-O3-NEXT:        Register Coalescer
; GCN-O3-NEXT:        Machine Block Frequency Analysis
; GCN-O3-NEXT:        Debug Variable Analysis
; GCN-O3-NEXT:        Live Stack Slot Analysis
; GCN-O3-NEXT:        Virtual Register Map
; GCN-O3-NEXT:        Live Register Matrix
; GCN-O3-NEXT:        Bundle Machine CFG Edges
; GCN-O3-NEXT:        Spill Code Placement Analysis
; GCN-O3-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O3-NEXT:        Machine Optimization Remark Emitter
; GCN-O3-NEXT:        Greedy Register Allocator
; GCN-O3-NEXT:        Virtual Register Rewriter
; GCN-O3-NEXT:        Stack Slot Coloring
; GCN-O3-NEXT:        SI lower SGPR spill instructions
; GCN-O3-NEXT:        Virtual Register Map
; GCN-O3-NEXT:        Live Register Matrix
; GCN-O3-NEXT:        SI Pre-allocate WWM Registers
; GCN-O3-NEXT:        Live Stack Slot Analysis
; GCN-O3-NEXT:        Greedy Register Allocator
; GCN-O3-NEXT:        SI Lower WWM Copies
; GCN-O3-NEXT:        Virtual Register Rewriter
; GCN-O3-NEXT:        AMDGPU Mark Last Scratch Load
; GCN-O3-NEXT:        Stack Slot Coloring
; GCN-O3-NEXT:        Machine Copy Propagation Pass
; GCN-O3-NEXT:        Machine Loop Invariant Code Motion
; GCN-O3-NEXT:        SI Fix VGPR copies
; GCN-O3-NEXT:        SI optimize exec mask operations
; GCN-O3-NEXT:        Remove Redundant DEBUG_VALUE analysis
; GCN-O3-NEXT:        Fixup Statepoint Caller Saved
; GCN-O3-NEXT:        PostRA Machine Sink
; GCN-O3-NEXT:        Machine Block Frequency Analysis
; GCN-O3-NEXT:        MachineDominator Tree Construction
; GCN-O3-NEXT:        MachinePostDominator Tree Construction
; GCN-O3-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O3-NEXT:        Machine Optimization Remark Emitter
; GCN-O3-NEXT:        Shrink Wrapping analysis
; GCN-O3-NEXT:        Prologue/Epilogue Insertion & Frame Finalization
; GCN-O3-NEXT:        Machine Late Instructions Cleanup Pass
; GCN-O3-NEXT:        Control Flow Optimizer
; GCN-O3-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O3-NEXT:        Tail Duplication
; GCN-O3-NEXT:        Machine Copy Propagation Pass
; GCN-O3-NEXT:        Post-RA pseudo instruction expansion pass
; GCN-O3-NEXT:        SI Shrink Instructions
; GCN-O3-NEXT:        SI post-RA bundler
; GCN-O3-NEXT:        MachineDominator Tree Construction
; GCN-O3-NEXT:        Machine Natural Loop Construction
; GCN-O3-NEXT:        PostRA Machine Instruction Scheduler
; GCN-O3-NEXT:        Machine Block Frequency Analysis
; GCN-O3-NEXT:        MachinePostDominator Tree Construction
; GCN-O3-NEXT:        Branch Probability Basic Block Placement
; GCN-O3-NEXT:        Insert fentry calls
; GCN-O3-NEXT:        Insert XRay ops
; GCN-O3-NEXT:        GCN Create VOPD Instructions
; GCN-O3-NEXT:        SI Memory Legalizer
; GCN-O3-NEXT:        MachineDominator Tree Construction
; GCN-O3-NEXT:        Machine Natural Loop Construction
; GCN-O3-NEXT:        MachinePostDominator Tree Construction
; GCN-O3-NEXT:        SI insert wait instructions
; GCN-O3-NEXT:        Insert required mode register values
; GCN-O3-NEXT:        SI Insert Hard Clauses
; GCN-O3-NEXT:        SI Final Branch Preparation
; GCN-O3-NEXT:        SI peephole optimizations
; GCN-O3-NEXT:        Post RA hazard recognizer
; GCN-O3-NEXT:        AMDGPU Insert waits for SGPR read hazards
; GCN-O3-NEXT:        AMDGPU Insert Delay ALU
; GCN-O3-NEXT:        Branch relaxation pass
; GCN-O3-NEXT:        AMDGPU Preload Kernel Arguments Prolog
; GCN-O3-NEXT:        Register Usage Information Collector Pass
; GCN-O3-NEXT:        Remove Loads Into Fake Uses
; GCN-O3-NEXT:        Live DEBUG_VALUE analysis
; GCN-O3-NEXT:        Machine Sanitizer Binary Metadata
; GCN-O3-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O3-NEXT:        Machine Optimization Remark Emitter
; GCN-O3-NEXT:        Stack Frame Layout Analysis
; GCN-O3-NEXT:        Function register usage analysis
; GCN-O3-NEXT:        AMDGPU Assembly Printer
; GCN-O3-NEXT:        Free MachineFunction
