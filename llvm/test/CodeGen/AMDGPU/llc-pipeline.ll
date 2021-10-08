; When EXPENSIVE_CHECKS are enabled, the machine verifier appears between each
; pass. Ignore it with 'grep -v'.
; fixme: the following line is added to cleanup bots, will be removed in weeks.
; RUN: rm -f %S/llc-pipeline.s
; RUN: llc -O0 -mtriple=amdgcn--amdhsa -disable-verify -debug-pass=Structure < %s 2>&1 \
; RUN:   | grep -v 'Verify generated machine code' | FileCheck -match-full-lines -strict-whitespace -check-prefix=GCN-O0 %s
; RUN: llc -O1 -mtriple=amdgcn--amdhsa -disable-verify -debug-pass=Structure < %s 2>&1 \
; RUN:   | grep -v 'Verify generated machine code' | FileCheck -match-full-lines -strict-whitespace -check-prefix=GCN-O1 %s
; RUN: llc -O1 -mtriple=amdgcn--amdhsa -disable-verify -amdgpu-scalar-ir-passes -amdgpu-sdwa-peephole \
; RUN:   -amdgpu-load-store-vectorizer -amdgpu-enable-pre-ra-optimizations -debug-pass=Structure < %s 2>&1 \
; RUN:   | grep -v 'Verify generated machine code' | FileCheck -match-full-lines -strict-whitespace -check-prefix=GCN-O1-OPTS %s
; RUN: llc -O2 -mtriple=amdgcn--amdhsa -disable-verify -debug-pass=Structure < %s 2>&1 \
; RUN:   | grep -v 'Verify generated machine code' | FileCheck -match-full-lines -strict-whitespace -check-prefix=GCN-O2 %s
; RUN: llc -O3 -mtriple=amdgcn--amdhsa -disable-verify -debug-pass=Structure < %s 2>&1 \
; RUN:   | grep -v 'Verify generated machine code' | FileCheck -match-full-lines -strict-whitespace -check-prefix=GCN-O3 %s

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
; GCN-O0-NEXT:    Pre-ISel Intrinsic Lowering
; GCN-O0-NEXT:    AMDGPU Printf lowering
; GCN-O0-NEXT:      FunctionPass Manager
; GCN-O0-NEXT:        Dominator Tree Construction
; GCN-O0-NEXT:    Lower ctors and dtors for AMDGPU
; GCN-O0-NEXT:    Fix function bitcasts for AMDGPU
; GCN-O0-NEXT:    AMDGPU Lower Kernel Calls
; GCN-O0-NEXT:    FunctionPass Manager
; GCN-O0-NEXT:      Early propagate attributes from kernels to functions
; GCN-O0-NEXT:    AMDGPU Lower Intrinsics
; GCN-O0-NEXT:    AMDGPU Inline All Functions
; GCN-O0-NEXT:    CallGraph Construction
; GCN-O0-NEXT:    Call Graph SCC Pass Manager
; GCN-O0-NEXT:      Inliner for always_inline functions
; GCN-O0-NEXT:    A No-Op Barrier Pass
; GCN-O0-NEXT:    Lower OpenCL enqueued blocks
; GCN-O0-NEXT:    Lower uses of LDS variables from non-kernel functions
; GCN-O0-NEXT:    FunctionPass Manager
; GCN-O0-NEXT:      Expand Atomic instructions
; GCN-O0-NEXT:      Lower constant intrinsics
; GCN-O0-NEXT:      Remove unreachable blocks from the CFG
; GCN-O0-NEXT:      Expand vector predication intrinsics
; GCN-O0-NEXT:      Scalarize Masked Memory Intrinsics
; GCN-O0-NEXT:      Expand reduction intrinsics
; GCN-O0-NEXT:    CallGraph Construction
; GCN-O0-NEXT:    Call Graph SCC Pass Manager
; GCN-O0-NEXT:      AMDGPU Annotate Kernel Features
; GCN-O0-NEXT:      FunctionPass Manager
; GCN-O0-NEXT:        AMDGPU Lower Kernel Arguments
; GCN-O0-NEXT:        Lazy Value Information Analysis
; GCN-O0-NEXT:        Lower SwitchInst's to branches
; GCN-O0-NEXT:        Lower invoke and unwind, for unwindless code generators
; GCN-O0-NEXT:        Remove unreachable blocks from the CFG
; GCN-O0-NEXT:        Post-Dominator Tree Construction
; GCN-O0-NEXT:        Dominator Tree Construction
; GCN-O0-NEXT:        Natural Loop Information
; GCN-O0-NEXT:        Legacy Divergence Analysis
; GCN-O0-NEXT:        Unify divergent function exit nodes
; GCN-O0-NEXT:        Lazy Value Information Analysis
; GCN-O0-NEXT:        Lower SwitchInst's to branches
; GCN-O0-NEXT:        Dominator Tree Construction
; GCN-O0-NEXT:        Natural Loop Information
; GCN-O0-NEXT:        Convert irreducible control-flow into natural loops
; GCN-O0-NEXT:        Fixup each natural loop to have a single exit block
; GCN-O0-NEXT:        Post-Dominator Tree Construction
; GCN-O0-NEXT:        Dominance Frontier Construction
; GCN-O0-NEXT:        Detect single entry single exit regions
; GCN-O0-NEXT:        Region Pass Manager
; GCN-O0-NEXT:          Structurize control flow
; GCN-O0-NEXT:        Post-Dominator Tree Construction
; GCN-O0-NEXT:        Natural Loop Information
; GCN-O0-NEXT:        Legacy Divergence Analysis
; GCN-O0-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O0-NEXT:        Function Alias Analysis Results
; GCN-O0-NEXT:        Memory SSA
; GCN-O0-NEXT:        AMDGPU Annotate Uniform Values
; GCN-O0-NEXT:        SI annotate control flow
; GCN-O0-NEXT:        LCSSA Verifier
; GCN-O0-NEXT:        Loop-Closed SSA Form Pass
; GCN-O0-NEXT:      DummyCGSCCPass
; GCN-O0-NEXT:      FunctionPass Manager
; GCN-O0-NEXT:        Safe Stack instrumentation pass
; GCN-O0-NEXT:        Insert stack protectors
; GCN-O0-NEXT:        Dominator Tree Construction
; GCN-O0-NEXT:        Post-Dominator Tree Construction
; GCN-O0-NEXT:        Natural Loop Information
; GCN-O0-NEXT:        Legacy Divergence Analysis
; GCN-O0-NEXT:        AMDGPU DAG->DAG Pattern Instruction Selection
; GCN-O0-NEXT:        MachineDominator Tree Construction
; GCN-O0-NEXT:        SI Fix SGPR copies
; GCN-O0-NEXT:        MachinePostDominator Tree Construction
; GCN-O0-NEXT:        SI Lower i1 Copies
; GCN-O0-NEXT:        Finalize ISel and expand pseudo-instructions
; GCN-O0-NEXT:        Local Stack Slot Allocation
; GCN-O0-NEXT:        Register Usage Information Propagation
; GCN-O0-NEXT:        Eliminate PHI nodes for register allocation
; GCN-O0-NEXT:        SI Lower control flow pseudo instructions
; GCN-O0-NEXT:        Two-Address instruction pass
; GCN-O0-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O0-NEXT:        Function Alias Analysis Results
; GCN-O0-NEXT:        MachineDominator Tree Construction
; GCN-O0-NEXT:        Slot index numbering
; GCN-O0-NEXT:        Live Interval Analysis
; GCN-O0-NEXT:        MachinePostDominator Tree Construction
; GCN-O0-NEXT:        SI Whole Quad Mode
; GCN-O0-NEXT:        Virtual Register Map
; GCN-O0-NEXT:        Live Register Matrix
; GCN-O0-NEXT:        SI Pre-allocate WWM Registers
; GCN-O0-NEXT:        Fast Register Allocator
; GCN-O0-NEXT:        SI lower SGPR spill instructions
; GCN-O0-NEXT:        Fast Register Allocator
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
; GCN-O0-NEXT:        MachinePostDominator Tree Construction
; GCN-O0-NEXT:        SI insert wait instructions
; GCN-O0-NEXT:        Insert required mode register values
; GCN-O0-NEXT:        MachineDominator Tree Construction
; GCN-O0-NEXT:        SI Final Branch Preparation
; GCN-O0-NEXT:        Post RA hazard recognizer
; GCN-O0-NEXT:        Branch relaxation pass
; GCN-O0-NEXT:        Register Usage Information Collector Pass
; GCN-O0-NEXT:        Live DEBUG_VALUE analysis
; GCN-O0-NEXT:      Function register usage analysis
; GCN-O0-NEXT:      FunctionPass Manager
; GCN-O0-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O0-NEXT:        Machine Optimization Remark Emitter
; GCN-O0-NEXT:        AMDGPU Assembly Printer
; GCN-O0-NEXT:        Free MachineFunction
; GCN-O0-NEXT:Pass Arguments:  -domtree
; GCN-O0-NEXT:  FunctionPass Manager
; GCN-O0-NEXT:    Dominator Tree Construction

; GCN-O1:Target Library Information
; GCN-O1-NEXT:Target Pass Configuration
; GCN-O1-NEXT:Machine Module Information
; GCN-O1-NEXT:Target Transform Information
; GCN-O1-NEXT:Assumption Cache Tracker
; GCN-O1-NEXT:Profile summary info
; GCN-O1-NEXT:AMDGPU Address space based Alias Analysis
; GCN-O1-NEXT:External Alias Analysis
; GCN-O1-NEXT:Type-Based Alias Analysis
; GCN-O1-NEXT:Scoped NoAlias Alias Analysis
; GCN-O1-NEXT:Argument Register Usage Information Storage
; GCN-O1-NEXT:Create Garbage Collector Module Metadata
; GCN-O1-NEXT:Machine Branch Probability Analysis
; GCN-O1-NEXT:Register Usage Information Storage
; GCN-O1-NEXT:  ModulePass Manager
; GCN-O1-NEXT:    Pre-ISel Intrinsic Lowering
; GCN-O1-NEXT:    AMDGPU Printf lowering
; GCN-O1-NEXT:      FunctionPass Manager
; GCN-O1-NEXT:        Dominator Tree Construction
; GCN-O1-NEXT:    Lower ctors and dtors for AMDGPU
; GCN-O1-NEXT:    Fix function bitcasts for AMDGPU
; GCN-O1-NEXT:    AMDGPU Lower Kernel Calls
; GCN-O1-NEXT:    FunctionPass Manager
; GCN-O1-NEXT:      Early propagate attributes from kernels to functions
; GCN-O1-NEXT:    AMDGPU Lower Intrinsics
; GCN-O1-NEXT:    AMDGPU Inline All Functions
; GCN-O1-NEXT:    CallGraph Construction
; GCN-O1-NEXT:    Call Graph SCC Pass Manager
; GCN-O1-NEXT:      Inliner for always_inline functions
; GCN-O1-NEXT:    A No-Op Barrier Pass
; GCN-O1-NEXT:    Lower OpenCL enqueued blocks
; GCN-O1-NEXT:    Lower uses of LDS variables from non-kernel functions
; GCN-O1-NEXT:    FunctionPass Manager
; GCN-O1-NEXT:      Infer address spaces
; GCN-O1-NEXT:      Expand Atomic instructions
; GCN-O1-NEXT:      AMDGPU Promote Alloca
; GCN-O1-NEXT:      Dominator Tree Construction
; GCN-O1-NEXT:      SROA
; GCN-O1-NEXT:      Post-Dominator Tree Construction
; GCN-O1-NEXT:      Natural Loop Information
; GCN-O1-NEXT:      Legacy Divergence Analysis
; GCN-O1-NEXT:      AMDGPU IR optimizations
; GCN-O1-NEXT:      Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:      Canonicalize natural loops
; GCN-O1-NEXT:      Scalar Evolution Analysis
; GCN-O1-NEXT:      Loop Pass Manager
; GCN-O1-NEXT:        Canonicalize Freeze Instructions in Loops
; GCN-O1-NEXT:        Induction Variable Users
; GCN-O1-NEXT:        Loop Strength Reduction
; GCN-O1-NEXT:      Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:      Function Alias Analysis Results
; GCN-O1-NEXT:      Merge contiguous icmps into a memcmp
; GCN-O1-NEXT:      Natural Loop Information
; GCN-O1-NEXT:      Lazy Branch Probability Analysis
; GCN-O1-NEXT:      Lazy Block Frequency Analysis
; GCN-O1-NEXT:      Expand memcmp() to load/stores
; GCN-O1-NEXT:      Lower constant intrinsics
; GCN-O1-NEXT:      Remove unreachable blocks from the CFG
; GCN-O1-NEXT:      Natural Loop Information
; GCN-O1-NEXT:      Post-Dominator Tree Construction
; GCN-O1-NEXT:      Branch Probability Analysis
; GCN-O1-NEXT:      Block Frequency Analysis
; GCN-O1-NEXT:      Constant Hoisting
; GCN-O1-NEXT:      Replace intrinsics with calls to vector library
; GCN-O1-NEXT:      Partially inline calls to library functions
; GCN-O1-NEXT:      Expand vector predication intrinsics
; GCN-O1-NEXT:      Scalarize Masked Memory Intrinsics
; GCN-O1-NEXT:      Expand reduction intrinsics
; GCN-O1-NEXT:    CallGraph Construction
; GCN-O1-NEXT:    Call Graph SCC Pass Manager
; GCN-O1-NEXT:      AMDGPU Annotate Kernel Features
; GCN-O1-NEXT:      FunctionPass Manager
; GCN-O1-NEXT:        AMDGPU Lower Kernel Arguments
; GCN-O1-NEXT:        Dominator Tree Construction
; GCN-O1-NEXT:        Natural Loop Information
; GCN-O1-NEXT:        CodeGen Prepare
; GCN-O1-NEXT:        Lazy Value Information Analysis
; GCN-O1-NEXT:        Lower SwitchInst's to branches
; GCN-O1-NEXT:        Lower invoke and unwind, for unwindless code generators
; GCN-O1-NEXT:        Remove unreachable blocks from the CFG
; GCN-O1-NEXT:        Dominator Tree Construction
; GCN-O1-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:        Function Alias Analysis Results
; GCN-O1-NEXT:        Flatten the CFG
; GCN-O1-NEXT:        Dominator Tree Construction
; GCN-O1-NEXT:        Post-Dominator Tree Construction
; GCN-O1-NEXT:        Natural Loop Information
; GCN-O1-NEXT:        Legacy Divergence Analysis
; GCN-O1-NEXT:        AMDGPU IR late optimizations
; GCN-O1-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:        Function Alias Analysis Results
; GCN-O1-NEXT:        Code sinking
; GCN-O1-NEXT:        Legacy Divergence Analysis
; GCN-O1-NEXT:        Unify divergent function exit nodes
; GCN-O1-NEXT:        Lazy Value Information Analysis
; GCN-O1-NEXT:        Lower SwitchInst's to branches
; GCN-O1-NEXT:        Dominator Tree Construction
; GCN-O1-NEXT:        Natural Loop Information
; GCN-O1-NEXT:        Convert irreducible control-flow into natural loops
; GCN-O1-NEXT:        Fixup each natural loop to have a single exit block
; GCN-O1-NEXT:        Post-Dominator Tree Construction
; GCN-O1-NEXT:        Dominance Frontier Construction
; GCN-O1-NEXT:        Detect single entry single exit regions
; GCN-O1-NEXT:        Region Pass Manager
; GCN-O1-NEXT:          Structurize control flow
; GCN-O1-NEXT:        Post-Dominator Tree Construction
; GCN-O1-NEXT:        Natural Loop Information
; GCN-O1-NEXT:        Legacy Divergence Analysis
; GCN-O1-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:        Function Alias Analysis Results
; GCN-O1-NEXT:        Memory SSA
; GCN-O1-NEXT:        AMDGPU Annotate Uniform Values
; GCN-O1-NEXT:        SI annotate control flow
; GCN-O1-NEXT:        LCSSA Verifier
; GCN-O1-NEXT:        Loop-Closed SSA Form Pass
; GCN-O1-NEXT:      DummyCGSCCPass
; GCN-O1-NEXT:      FunctionPass Manager
; GCN-O1-NEXT:        Safe Stack instrumentation pass
; GCN-O1-NEXT:        Insert stack protectors
; GCN-O1-NEXT:        Dominator Tree Construction
; GCN-O1-NEXT:        Post-Dominator Tree Construction
; GCN-O1-NEXT:        Natural Loop Information
; GCN-O1-NEXT:        Legacy Divergence Analysis
; GCN-O1-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O1-NEXT:        Function Alias Analysis Results
; GCN-O1-NEXT:        Branch Probability Analysis
; GCN-O1-NEXT:        Lazy Branch Probability Analysis
; GCN-O1-NEXT:        Lazy Block Frequency Analysis
; GCN-O1-NEXT:        AMDGPU DAG->DAG Pattern Instruction Selection
; GCN-O1-NEXT:        MachineDominator Tree Construction
; GCN-O1-NEXT:        SI Fix SGPR copies
; GCN-O1-NEXT:        MachinePostDominator Tree Construction
; GCN-O1-NEXT:        SI Lower i1 Copies
; GCN-O1-NEXT:        Finalize ISel and expand pseudo-instructions
; GCN-O1-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O1-NEXT:        Early Tail Duplication
; GCN-O1-NEXT:        Optimize machine instruction PHIs
; GCN-O1-NEXT:        Slot index numbering
; GCN-O1-NEXT:        Merge disjoint stack slots
; GCN-O1-NEXT:        Local Stack Slot Allocation
; GCN-O1-NEXT:        Remove dead machine instructions
; GCN-O1-NEXT:        MachineDominator Tree Construction
; GCN-O1-NEXT:        Machine Natural Loop Construction
; GCN-O1-NEXT:        Machine Block Frequency Analysis
; GCN-O1-NEXT:        Early Machine Loop Invariant Code Motion
; GCN-O1-NEXT:        MachineDominator Tree Construction
; GCN-O1-NEXT:        Machine Block Frequency Analysis
; GCN-O1-NEXT:        Machine Common Subexpression Elimination
; GCN-O1-NEXT:        MachinePostDominator Tree Construction
; GCN-O1-NEXT:        Machine code sinking
; GCN-O1-NEXT:        Peephole Optimizations
; GCN-O1-NEXT:        Remove dead machine instructions
; GCN-O1-NEXT:        SI Fold Operands
; GCN-O1-NEXT:        GCN DPP Combine
; GCN-O1-NEXT:        SI Load Store Optimizer
; GCN-O1-NEXT:        Remove dead machine instructions
; GCN-O1-NEXT:        SI Shrink Instructions
; GCN-O1-NEXT:        Register Usage Information Propagation
; GCN-O1-NEXT:        Detect Dead Lanes
; GCN-O1-NEXT:        Remove dead machine instructions
; GCN-O1-NEXT:        Process Implicit Definitions
; GCN-O1-NEXT:        Remove unreachable machine basic blocks
; GCN-O1-NEXT:        Live Variable Analysis
; GCN-O1-NEXT:        MachineDominator Tree Construction
; GCN-O1-NEXT:        SI Optimize VGPR LiveRange
; GCN-O1-NEXT:        Eliminate PHI nodes for register allocation
; GCN-O1-NEXT:        SI Lower control flow pseudo instructions
; GCN-O1-NEXT:        Two-Address instruction pass
; GCN-O1-NEXT:        Slot index numbering
; GCN-O1-NEXT:        Live Interval Analysis
; GCN-O1-NEXT:        Machine Natural Loop Construction
; GCN-O1-NEXT:        Simple Register Coalescing
; GCN-O1-NEXT:        Rename Disconnected Subregister Components
; GCN-O1-NEXT:        Machine Instruction Scheduler
; GCN-O1-NEXT:        MachinePostDominator Tree Construction
; GCN-O1-NEXT:        SI Whole Quad Mode
; GCN-O1-NEXT:        Virtual Register Map
; GCN-O1-NEXT:        Live Register Matrix
; GCN-O1-NEXT:        SI Pre-allocate WWM Registers
; GCN-O1-NEXT:        SI optimize exec mask operations pre-RA
; GCN-O1-NEXT:        Machine Natural Loop Construction
; GCN-O1-NEXT:        Machine Block Frequency Analysis
; GCN-O1-NEXT:        Debug Variable Analysis
; GCN-O1-NEXT:        Live Stack Slot Analysis
; GCN-O1-NEXT:        Virtual Register Map
; GCN-O1-NEXT:        Live Register Matrix
; GCN-O1-NEXT:        Bundle Machine CFG Edges
; GCN-O1-NEXT:        Spill Code Placement Analysis
; GCN-O1-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O1-NEXT:        Machine Optimization Remark Emitter
; GCN-O1-NEXT:        Greedy Register Allocator
; GCN-O1-NEXT:        Virtual Register Rewriter
; GCN-O1-NEXT:        SI lower SGPR spill instructions
; GCN-O1-NEXT:        Virtual Register Map
; GCN-O1-NEXT:        Live Register Matrix
; GCN-O1-NEXT:        Greedy Register Allocator
; GCN-O1-NEXT:        GCN NSA Reassign
; GCN-O1-NEXT:        Virtual Register Rewriter
; GCN-O1-NEXT:        Stack Slot Coloring
; GCN-O1-NEXT:        Machine Copy Propagation Pass
; GCN-O1-NEXT:        Machine Loop Invariant Code Motion
; GCN-O1-NEXT:        SI Fix VGPR copies
; GCN-O1-NEXT:        SI optimize exec mask operations
; GCN-O1-NEXT:        Remove Redundant DEBUG_VALUE analysis
; GCN-O1-NEXT:        Fixup Statepoint Caller Saved
; GCN-O1-NEXT:        PostRA Machine Sink
; GCN-O1-NEXT:        MachineDominator Tree Construction
; GCN-O1-NEXT:        Machine Natural Loop Construction
; GCN-O1-NEXT:        Machine Block Frequency Analysis
; GCN-O1-NEXT:        MachinePostDominator Tree Construction
; GCN-O1-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O1-NEXT:        Machine Optimization Remark Emitter
; GCN-O1-NEXT:        Shrink Wrapping analysis
; GCN-O1-NEXT:        Prologue/Epilogue Insertion & Frame Finalization
; GCN-O1-NEXT:        Control Flow Optimizer
; GCN-O1-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O1-NEXT:        Tail Duplication
; GCN-O1-NEXT:        Machine Copy Propagation Pass
; GCN-O1-NEXT:        Post-RA pseudo instruction expansion pass
; GCN-O1-NEXT:        SI post-RA bundler
; GCN-O1-NEXT:        MachineDominator Tree Construction
; GCN-O1-NEXT:        Machine Natural Loop Construction
; GCN-O1-NEXT:        PostRA Machine Instruction Scheduler
; GCN-O1-NEXT:        Machine Block Frequency Analysis
; GCN-O1-NEXT:        MachinePostDominator Tree Construction
; GCN-O1-NEXT:        Branch Probability Basic Block Placement
; GCN-O1-NEXT:        Insert fentry calls
; GCN-O1-NEXT:        Insert XRay ops
; GCN-O1-NEXT:        SI Memory Legalizer
; GCN-O1-NEXT:        MachinePostDominator Tree Construction
; GCN-O1-NEXT:        SI insert wait instructions
; GCN-O1-NEXT:        SI Shrink Instructions
; GCN-O1-NEXT:        Insert required mode register values
; GCN-O1-NEXT:        SI Insert Hard Clauses
; GCN-O1-NEXT:        MachineDominator Tree Construction
; GCN-O1-NEXT:        SI Final Branch Preparation
; GCN-O1-NEXT:        SI peephole optimizations
; GCN-O1-NEXT:        Post RA hazard recognizer
; GCN-O1-NEXT:        Branch relaxation pass
; GCN-O1-NEXT:        Register Usage Information Collector Pass
; GCN-O1-NEXT:        Live DEBUG_VALUE analysis
; GCN-O1-NEXT:      Function register usage analysis
; GCN-O1-NEXT:      FunctionPass Manager
; GCN-O1-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O1-NEXT:        Machine Optimization Remark Emitter
; GCN-O1-NEXT:        AMDGPU Assembly Printer
; GCN-O1-NEXT:        Free MachineFunction
; GCN-O1-NEXT:Pass Arguments:  -domtree
; GCN-O1-NEXT:  FunctionPass Manager
; GCN-O1-NEXT:    Dominator Tree Construction

; GCN-O1-OPTS:Target Library Information
; GCN-O1-OPTS-NEXT:Target Pass Configuration
; GCN-O1-OPTS-NEXT:Machine Module Information
; GCN-O1-OPTS-NEXT:Target Transform Information
; GCN-O1-OPTS-NEXT:Assumption Cache Tracker
; GCN-O1-OPTS-NEXT:Profile summary info
; GCN-O1-OPTS-NEXT:AMDGPU Address space based Alias Analysis
; GCN-O1-OPTS-NEXT:External Alias Analysis
; GCN-O1-OPTS-NEXT:Type-Based Alias Analysis
; GCN-O1-OPTS-NEXT:Scoped NoAlias Alias Analysis
; GCN-O1-OPTS-NEXT:Argument Register Usage Information Storage
; GCN-O1-OPTS-NEXT:Create Garbage Collector Module Metadata
; GCN-O1-OPTS-NEXT:Machine Branch Probability Analysis
; GCN-O1-OPTS-NEXT:Register Usage Information Storage
; GCN-O1-OPTS-NEXT:  ModulePass Manager
; GCN-O1-OPTS-NEXT:    Pre-ISel Intrinsic Lowering
; GCN-O1-OPTS-NEXT:    AMDGPU Printf lowering
; GCN-O1-OPTS-NEXT:      FunctionPass Manager
; GCN-O1-OPTS-NEXT:        Dominator Tree Construction
; GCN-O1-OPTS-NEXT:    Lower ctors and dtors for AMDGPU
; GCN-O1-OPTS-NEXT:    Fix function bitcasts for AMDGPU
; GCN-O1-OPTS-NEXT:    AMDGPU Lower Kernel Calls
; GCN-O1-OPTS-NEXT:    FunctionPass Manager
; GCN-O1-OPTS-NEXT:      Early propagate attributes from kernels to functions
; GCN-O1-OPTS-NEXT:    AMDGPU Lower Intrinsics
; GCN-O1-OPTS-NEXT:    AMDGPU Inline All Functions
; GCN-O1-OPTS-NEXT:    CallGraph Construction
; GCN-O1-OPTS-NEXT:    Call Graph SCC Pass Manager
; GCN-O1-OPTS-NEXT:      Inliner for always_inline functions
; GCN-O1-OPTS-NEXT:    A No-Op Barrier Pass
; GCN-O1-OPTS-NEXT:    Lower OpenCL enqueued blocks
; GCN-O1-OPTS-NEXT:    Lower uses of LDS variables from non-kernel functions
; GCN-O1-OPTS-NEXT:    FunctionPass Manager
; GCN-O1-OPTS-NEXT:      Infer address spaces
; GCN-O1-OPTS-NEXT:      Expand Atomic instructions
; GCN-O1-OPTS-NEXT:      AMDGPU Promote Alloca
; GCN-O1-OPTS-NEXT:      Dominator Tree Construction
; GCN-O1-OPTS-NEXT:      SROA
; GCN-O1-OPTS-NEXT:      Basic Alias Analysis (stateless AA impl)
; GCN-O1-OPTS-NEXT:      Function Alias Analysis Results
; GCN-O1-OPTS-NEXT:      Memory SSA
; GCN-O1-OPTS-NEXT:      Natural Loop Information
; GCN-O1-OPTS-NEXT:      Canonicalize natural loops
; GCN-O1-OPTS-NEXT:      LCSSA Verifier
; GCN-O1-OPTS-NEXT:      Loop-Closed SSA Form Pass
; GCN-O1-OPTS-NEXT:      Scalar Evolution Analysis
; GCN-O1-OPTS-NEXT:      Lazy Branch Probability Analysis
; GCN-O1-OPTS-NEXT:      Lazy Block Frequency Analysis
; GCN-O1-OPTS-NEXT:      Loop Pass Manager
; GCN-O1-OPTS-NEXT:        Loop Invariant Code Motion
; GCN-O1-OPTS-NEXT:      Split GEPs to a variadic base and a constant offset for better CSE
; GCN-O1-OPTS-NEXT:      Speculatively execute instructions
; GCN-O1-OPTS-NEXT:      Scalar Evolution Analysis
; GCN-O1-OPTS-NEXT:      Straight line strength reduction
; GCN-O1-OPTS-NEXT:      Early CSE
; GCN-O1-OPTS-NEXT:      Scalar Evolution Analysis
; GCN-O1-OPTS-NEXT:      Nary reassociation
; GCN-O1-OPTS-NEXT:      Early CSE
; GCN-O1-OPTS-NEXT:      Post-Dominator Tree Construction
; GCN-O1-OPTS-NEXT:      Legacy Divergence Analysis
; GCN-O1-OPTS-NEXT:      AMDGPU IR optimizations
; GCN-O1-OPTS-NEXT:      Canonicalize natural loops
; GCN-O1-OPTS-NEXT:      Scalar Evolution Analysis
; GCN-O1-OPTS-NEXT:      Loop Pass Manager
; GCN-O1-OPTS-NEXT:        Canonicalize Freeze Instructions in Loops
; GCN-O1-OPTS-NEXT:        Induction Variable Users
; GCN-O1-OPTS-NEXT:        Loop Strength Reduction
; GCN-O1-OPTS-NEXT:      Basic Alias Analysis (stateless AA impl)
; GCN-O1-OPTS-NEXT:      Function Alias Analysis Results
; GCN-O1-OPTS-NEXT:      Merge contiguous icmps into a memcmp
; GCN-O1-OPTS-NEXT:      Natural Loop Information
; GCN-O1-OPTS-NEXT:      Lazy Branch Probability Analysis
; GCN-O1-OPTS-NEXT:      Lazy Block Frequency Analysis
; GCN-O1-OPTS-NEXT:      Expand memcmp() to load/stores
; GCN-O1-OPTS-NEXT:      Lower constant intrinsics
; GCN-O1-OPTS-NEXT:      Remove unreachable blocks from the CFG
; GCN-O1-OPTS-NEXT:      Natural Loop Information
; GCN-O1-OPTS-NEXT:      Post-Dominator Tree Construction
; GCN-O1-OPTS-NEXT:      Branch Probability Analysis
; GCN-O1-OPTS-NEXT:      Block Frequency Analysis
; GCN-O1-OPTS-NEXT:      Constant Hoisting
; GCN-O1-OPTS-NEXT:      Replace intrinsics with calls to vector library
; GCN-O1-OPTS-NEXT:      Partially inline calls to library functions
; GCN-O1-OPTS-NEXT:      Expand vector predication intrinsics
; GCN-O1-OPTS-NEXT:      Scalarize Masked Memory Intrinsics
; GCN-O1-OPTS-NEXT:      Expand reduction intrinsics
; GCN-O1-OPTS-NEXT:      Early CSE
; GCN-O1-OPTS-NEXT:    CallGraph Construction
; GCN-O1-OPTS-NEXT:    Call Graph SCC Pass Manager
; GCN-O1-OPTS-NEXT:      AMDGPU Annotate Kernel Features
; GCN-O1-OPTS-NEXT:      FunctionPass Manager
; GCN-O1-OPTS-NEXT:        AMDGPU Lower Kernel Arguments
; GCN-O1-OPTS-NEXT:        Dominator Tree Construction
; GCN-O1-OPTS-NEXT:        Natural Loop Information
; GCN-O1-OPTS-NEXT:        CodeGen Prepare
; GCN-O1-OPTS-NEXT:        Dominator Tree Construction
; GCN-O1-OPTS-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O1-OPTS-NEXT:        Function Alias Analysis Results
; GCN-O1-OPTS-NEXT:        Natural Loop Information
; GCN-O1-OPTS-NEXT:        Scalar Evolution Analysis
; GCN-O1-OPTS-NEXT:        GPU Load and Store Vectorizer
; GCN-O1-OPTS-NEXT:        Lazy Value Information Analysis
; GCN-O1-OPTS-NEXT:        Lower SwitchInst's to branches
; GCN-O1-OPTS-NEXT:        Lower invoke and unwind, for unwindless code generators
; GCN-O1-OPTS-NEXT:        Remove unreachable blocks from the CFG
; GCN-O1-OPTS-NEXT:        Dominator Tree Construction
; GCN-O1-OPTS-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O1-OPTS-NEXT:        Function Alias Analysis Results
; GCN-O1-OPTS-NEXT:        Flatten the CFG
; GCN-O1-OPTS-NEXT:        Dominator Tree Construction
; GCN-O1-OPTS-NEXT:        Post-Dominator Tree Construction
; GCN-O1-OPTS-NEXT:        Natural Loop Information
; GCN-O1-OPTS-NEXT:        Legacy Divergence Analysis
; GCN-O1-OPTS-NEXT:        AMDGPU IR late optimizations
; GCN-O1-OPTS-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O1-OPTS-NEXT:        Function Alias Analysis Results
; GCN-O1-OPTS-NEXT:        Code sinking
; GCN-O1-OPTS-NEXT:        Legacy Divergence Analysis
; GCN-O1-OPTS-NEXT:        Unify divergent function exit nodes
; GCN-O1-OPTS-NEXT:        Lazy Value Information Analysis
; GCN-O1-OPTS-NEXT:        Lower SwitchInst's to branches
; GCN-O1-OPTS-NEXT:        Dominator Tree Construction
; GCN-O1-OPTS-NEXT:        Natural Loop Information
; GCN-O1-OPTS-NEXT:        Convert irreducible control-flow into natural loops
; GCN-O1-OPTS-NEXT:        Fixup each natural loop to have a single exit block
; GCN-O1-OPTS-NEXT:        Post-Dominator Tree Construction
; GCN-O1-OPTS-NEXT:        Dominance Frontier Construction
; GCN-O1-OPTS-NEXT:        Detect single entry single exit regions
; GCN-O1-OPTS-NEXT:        Region Pass Manager
; GCN-O1-OPTS-NEXT:          Structurize control flow
; GCN-O1-OPTS-NEXT:        Post-Dominator Tree Construction
; GCN-O1-OPTS-NEXT:        Natural Loop Information
; GCN-O1-OPTS-NEXT:        Legacy Divergence Analysis
; GCN-O1-OPTS-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O1-OPTS-NEXT:        Function Alias Analysis Results
; GCN-O1-OPTS-NEXT:        Memory SSA
; GCN-O1-OPTS-NEXT:        AMDGPU Annotate Uniform Values
; GCN-O1-OPTS-NEXT:        SI annotate control flow
; GCN-O1-OPTS-NEXT:        LCSSA Verifier
; GCN-O1-OPTS-NEXT:        Loop-Closed SSA Form Pass
; GCN-O1-OPTS-NEXT:      DummyCGSCCPass
; GCN-O1-OPTS-NEXT:      FunctionPass Manager
; GCN-O1-OPTS-NEXT:        Safe Stack instrumentation pass
; GCN-O1-OPTS-NEXT:        Insert stack protectors
; GCN-O1-OPTS-NEXT:        Dominator Tree Construction
; GCN-O1-OPTS-NEXT:        Post-Dominator Tree Construction
; GCN-O1-OPTS-NEXT:        Natural Loop Information
; GCN-O1-OPTS-NEXT:        Legacy Divergence Analysis
; GCN-O1-OPTS-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O1-OPTS-NEXT:        Function Alias Analysis Results
; GCN-O1-OPTS-NEXT:        Branch Probability Analysis
; GCN-O1-OPTS-NEXT:        Lazy Branch Probability Analysis
; GCN-O1-OPTS-NEXT:        Lazy Block Frequency Analysis
; GCN-O1-OPTS-NEXT:        AMDGPU DAG->DAG Pattern Instruction Selection
; GCN-O1-OPTS-NEXT:        MachineDominator Tree Construction
; GCN-O1-OPTS-NEXT:        SI Fix SGPR copies
; GCN-O1-OPTS-NEXT:        MachinePostDominator Tree Construction
; GCN-O1-OPTS-NEXT:        SI Lower i1 Copies
; GCN-O1-OPTS-NEXT:        Finalize ISel and expand pseudo-instructions
; GCN-O1-OPTS-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O1-OPTS-NEXT:        Early Tail Duplication
; GCN-O1-OPTS-NEXT:        Optimize machine instruction PHIs
; GCN-O1-OPTS-NEXT:        Slot index numbering
; GCN-O1-OPTS-NEXT:        Merge disjoint stack slots
; GCN-O1-OPTS-NEXT:        Local Stack Slot Allocation
; GCN-O1-OPTS-NEXT:        Remove dead machine instructions
; GCN-O1-OPTS-NEXT:        MachineDominator Tree Construction
; GCN-O1-OPTS-NEXT:        Machine Natural Loop Construction
; GCN-O1-OPTS-NEXT:        Machine Block Frequency Analysis
; GCN-O1-OPTS-NEXT:        Early Machine Loop Invariant Code Motion
; GCN-O1-OPTS-NEXT:        MachineDominator Tree Construction
; GCN-O1-OPTS-NEXT:        Machine Block Frequency Analysis
; GCN-O1-OPTS-NEXT:        Machine Common Subexpression Elimination
; GCN-O1-OPTS-NEXT:        MachinePostDominator Tree Construction
; GCN-O1-OPTS-NEXT:        Machine code sinking
; GCN-O1-OPTS-NEXT:        Peephole Optimizations
; GCN-O1-OPTS-NEXT:        Remove dead machine instructions
; GCN-O1-OPTS-NEXT:        SI Fold Operands
; GCN-O1-OPTS-NEXT:        GCN DPP Combine
; GCN-O1-OPTS-NEXT:        SI Load Store Optimizer
; GCN-O1-OPTS-NEXT:        SI Peephole SDWA
; GCN-O1-OPTS-NEXT:        Machine Block Frequency Analysis
; GCN-O1-OPTS-NEXT:        MachineDominator Tree Construction
; GCN-O1-OPTS-NEXT:        Early Machine Loop Invariant Code Motion
; GCN-O1-OPTS-NEXT:        MachineDominator Tree Construction
; GCN-O1-OPTS-NEXT:        Machine Block Frequency Analysis
; GCN-O1-OPTS-NEXT:        Machine Common Subexpression Elimination
; GCN-O1-OPTS-NEXT:        SI Fold Operands
; GCN-O1-OPTS-NEXT:        Remove dead machine instructions
; GCN-O1-OPTS-NEXT:        SI Shrink Instructions
; GCN-O1-OPTS-NEXT:        Register Usage Information Propagation
; GCN-O1-OPTS-NEXT:        Detect Dead Lanes
; GCN-O1-OPTS-NEXT:        Remove dead machine instructions
; GCN-O1-OPTS-NEXT:        Process Implicit Definitions
; GCN-O1-OPTS-NEXT:        Remove unreachable machine basic blocks
; GCN-O1-OPTS-NEXT:        Live Variable Analysis
; GCN-O1-OPTS-NEXT:        SI Optimize VGPR LiveRange
; GCN-O1-OPTS-NEXT:        Eliminate PHI nodes for register allocation
; GCN-O1-OPTS-NEXT:        SI Lower control flow pseudo instructions
; GCN-O1-OPTS-NEXT:        Two-Address instruction pass
; GCN-O1-OPTS-NEXT:        Slot index numbering
; GCN-O1-OPTS-NEXT:        Live Interval Analysis
; GCN-O1-OPTS-NEXT:        Machine Natural Loop Construction
; GCN-O1-OPTS-NEXT:        Simple Register Coalescing
; GCN-O1-OPTS-NEXT:        Rename Disconnected Subregister Components
; GCN-O1-OPTS-NEXT:        AMDGPU Pre-RA optimizations
; GCN-O1-OPTS-NEXT:        Machine Instruction Scheduler
; GCN-O1-OPTS-NEXT:        MachinePostDominator Tree Construction
; GCN-O1-OPTS-NEXT:        SI Whole Quad Mode
; GCN-O1-OPTS-NEXT:        Virtual Register Map
; GCN-O1-OPTS-NEXT:        Live Register Matrix
; GCN-O1-OPTS-NEXT:        SI Pre-allocate WWM Registers
; GCN-O1-OPTS-NEXT:        SI optimize exec mask operations pre-RA
; GCN-O1-OPTS-NEXT:        Machine Natural Loop Construction
; GCN-O1-OPTS-NEXT:        Machine Block Frequency Analysis
; GCN-O1-OPTS-NEXT:        Debug Variable Analysis
; GCN-O1-OPTS-NEXT:        Live Stack Slot Analysis
; GCN-O1-OPTS-NEXT:        Virtual Register Map
; GCN-O1-OPTS-NEXT:        Live Register Matrix
; GCN-O1-OPTS-NEXT:        Bundle Machine CFG Edges
; GCN-O1-OPTS-NEXT:        Spill Code Placement Analysis
; GCN-O1-OPTS-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O1-OPTS-NEXT:        Machine Optimization Remark Emitter
; GCN-O1-OPTS-NEXT:        Greedy Register Allocator
; GCN-O1-OPTS-NEXT:        Virtual Register Rewriter
; GCN-O1-OPTS-NEXT:        SI lower SGPR spill instructions
; GCN-O1-OPTS-NEXT:        Virtual Register Map
; GCN-O1-OPTS-NEXT:        Live Register Matrix
; GCN-O1-OPTS-NEXT:        Greedy Register Allocator
; GCN-O1-OPTS-NEXT:        GCN NSA Reassign
; GCN-O1-OPTS-NEXT:        Virtual Register Rewriter
; GCN-O1-OPTS-NEXT:        Stack Slot Coloring
; GCN-O1-OPTS-NEXT:        Machine Copy Propagation Pass
; GCN-O1-OPTS-NEXT:        Machine Loop Invariant Code Motion
; GCN-O1-OPTS-NEXT:        SI Fix VGPR copies
; GCN-O1-OPTS-NEXT:        SI optimize exec mask operations
; GCN-O1-OPTS-NEXT:        Remove Redundant DEBUG_VALUE analysis
; GCN-O1-OPTS-NEXT:        Fixup Statepoint Caller Saved
; GCN-O1-OPTS-NEXT:        PostRA Machine Sink
; GCN-O1-OPTS-NEXT:        MachineDominator Tree Construction
; GCN-O1-OPTS-NEXT:        Machine Natural Loop Construction
; GCN-O1-OPTS-NEXT:        Machine Block Frequency Analysis
; GCN-O1-OPTS-NEXT:        MachinePostDominator Tree Construction
; GCN-O1-OPTS-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O1-OPTS-NEXT:        Machine Optimization Remark Emitter
; GCN-O1-OPTS-NEXT:        Shrink Wrapping analysis
; GCN-O1-OPTS-NEXT:        Prologue/Epilogue Insertion & Frame Finalization
; GCN-O1-OPTS-NEXT:        Control Flow Optimizer
; GCN-O1-OPTS-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O1-OPTS-NEXT:        Tail Duplication
; GCN-O1-OPTS-NEXT:        Machine Copy Propagation Pass
; GCN-O1-OPTS-NEXT:        Post-RA pseudo instruction expansion pass
; GCN-O1-OPTS-NEXT:        SI post-RA bundler
; GCN-O1-OPTS-NEXT:        MachineDominator Tree Construction
; GCN-O1-OPTS-NEXT:        Machine Natural Loop Construction
; GCN-O1-OPTS-NEXT:        PostRA Machine Instruction Scheduler
; GCN-O1-OPTS-NEXT:        Machine Block Frequency Analysis
; GCN-O1-OPTS-NEXT:        MachinePostDominator Tree Construction
; GCN-O1-OPTS-NEXT:        Branch Probability Basic Block Placement
; GCN-O1-OPTS-NEXT:        Insert fentry calls
; GCN-O1-OPTS-NEXT:        Insert XRay ops
; GCN-O1-OPTS-NEXT:        SI Memory Legalizer
; GCN-O1-OPTS-NEXT:        MachinePostDominator Tree Construction
; GCN-O1-OPTS-NEXT:        SI insert wait instructions
; GCN-O1-OPTS-NEXT:        SI Shrink Instructions
; GCN-O1-OPTS-NEXT:        Insert required mode register values
; GCN-O1-OPTS-NEXT:        SI Insert Hard Clauses
; GCN-O1-OPTS-NEXT:        MachineDominator Tree Construction
; GCN-O1-OPTS-NEXT:        SI Final Branch Preparation
; GCN-O1-OPTS-NEXT:        SI peephole optimizations
; GCN-O1-OPTS-NEXT:        Post RA hazard recognizer
; GCN-O1-OPTS-NEXT:        Branch relaxation pass
; GCN-O1-OPTS-NEXT:        Register Usage Information Collector Pass
; GCN-O1-OPTS-NEXT:        Live DEBUG_VALUE analysis
; GCN-O1-OPTS-NEXT:      Function register usage analysis
; GCN-O1-OPTS-NEXT:      FunctionPass Manager
; GCN-O1-OPTS-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O1-OPTS-NEXT:        Machine Optimization Remark Emitter
; GCN-O1-OPTS-NEXT:        AMDGPU Assembly Printer
; GCN-O1-OPTS-NEXT:        Free MachineFunction
; GCN-O1-OPTS-NEXT:Pass Arguments:  -domtree
; GCN-O1-OPTS-NEXT:  FunctionPass Manager
; GCN-O1-OPTS-NEXT:    Dominator Tree Construction

; GCN-O2:Target Library Information
; GCN-O2-NEXT:Target Pass Configuration
; GCN-O2-NEXT:Machine Module Information
; GCN-O2-NEXT:Target Transform Information
; GCN-O2-NEXT:Assumption Cache Tracker
; GCN-O2-NEXT:Profile summary info
; GCN-O2-NEXT:AMDGPU Address space based Alias Analysis
; GCN-O2-NEXT:External Alias Analysis
; GCN-O2-NEXT:Type-Based Alias Analysis
; GCN-O2-NEXT:Scoped NoAlias Alias Analysis
; GCN-O2-NEXT:Argument Register Usage Information Storage
; GCN-O2-NEXT:Create Garbage Collector Module Metadata
; GCN-O2-NEXT:Machine Branch Probability Analysis
; GCN-O2-NEXT:Register Usage Information Storage
; GCN-O2-NEXT:  ModulePass Manager
; GCN-O2-NEXT:    Pre-ISel Intrinsic Lowering
; GCN-O2-NEXT:    AMDGPU Printf lowering
; GCN-O2-NEXT:      FunctionPass Manager
; GCN-O2-NEXT:        Dominator Tree Construction
; GCN-O2-NEXT:    Lower ctors and dtors for AMDGPU
; GCN-O2-NEXT:    Fix function bitcasts for AMDGPU
; GCN-O2-NEXT:    AMDGPU Lower Kernel Calls
; GCN-O2-NEXT:    FunctionPass Manager
; GCN-O2-NEXT:      Early propagate attributes from kernels to functions
; GCN-O2-NEXT:    AMDGPU Lower Intrinsics
; GCN-O2-NEXT:    AMDGPU Inline All Functions
; GCN-O2-NEXT:    CallGraph Construction
; GCN-O2-NEXT:    Call Graph SCC Pass Manager
; GCN-O2-NEXT:      Inliner for always_inline functions
; GCN-O2-NEXT:    A No-Op Barrier Pass
; GCN-O2-NEXT:    Lower OpenCL enqueued blocks
; GCN-O2-NEXT:    Lower uses of LDS variables from non-kernel functions
; GCN-O2-NEXT:    FunctionPass Manager
; GCN-O2-NEXT:      Infer address spaces
; GCN-O2-NEXT:      Expand Atomic instructions
; GCN-O2-NEXT:      AMDGPU Promote Alloca
; GCN-O2-NEXT:      Dominator Tree Construction
; GCN-O2-NEXT:      SROA
; GCN-O2-NEXT:      Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:      Function Alias Analysis Results
; GCN-O2-NEXT:      Memory SSA
; GCN-O2-NEXT:      Natural Loop Information
; GCN-O2-NEXT:      Canonicalize natural loops
; GCN-O2-NEXT:      LCSSA Verifier
; GCN-O2-NEXT:      Loop-Closed SSA Form Pass
; GCN-O2-NEXT:      Scalar Evolution Analysis
; GCN-O2-NEXT:      Lazy Branch Probability Analysis
; GCN-O2-NEXT:      Lazy Block Frequency Analysis
; GCN-O2-NEXT:      Loop Pass Manager
; GCN-O2-NEXT:        Loop Invariant Code Motion
; GCN-O2-NEXT:      Split GEPs to a variadic base and a constant offset for better CSE
; GCN-O2-NEXT:      Speculatively execute instructions
; GCN-O2-NEXT:      Scalar Evolution Analysis
; GCN-O2-NEXT:      Straight line strength reduction
; GCN-O2-NEXT:      Early CSE
; GCN-O2-NEXT:      Scalar Evolution Analysis
; GCN-O2-NEXT:      Nary reassociation
; GCN-O2-NEXT:      Early CSE
; GCN-O2-NEXT:      Post-Dominator Tree Construction
; GCN-O2-NEXT:      Legacy Divergence Analysis
; GCN-O2-NEXT:      AMDGPU IR optimizations
; GCN-O2-NEXT:      Canonicalize natural loops
; GCN-O2-NEXT:      Scalar Evolution Analysis
; GCN-O2-NEXT:      Loop Pass Manager
; GCN-O2-NEXT:        Canonicalize Freeze Instructions in Loops
; GCN-O2-NEXT:        Induction Variable Users
; GCN-O2-NEXT:        Loop Strength Reduction
; GCN-O2-NEXT:      Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:      Function Alias Analysis Results
; GCN-O2-NEXT:      Merge contiguous icmps into a memcmp
; GCN-O2-NEXT:      Natural Loop Information
; GCN-O2-NEXT:      Lazy Branch Probability Analysis
; GCN-O2-NEXT:      Lazy Block Frequency Analysis
; GCN-O2-NEXT:      Expand memcmp() to load/stores
; GCN-O2-NEXT:      Lower constant intrinsics
; GCN-O2-NEXT:      Remove unreachable blocks from the CFG
; GCN-O2-NEXT:      Natural Loop Information
; GCN-O2-NEXT:      Post-Dominator Tree Construction
; GCN-O2-NEXT:      Branch Probability Analysis
; GCN-O2-NEXT:      Block Frequency Analysis
; GCN-O2-NEXT:      Constant Hoisting
; GCN-O2-NEXT:      Replace intrinsics with calls to vector library
; GCN-O2-NEXT:      Partially inline calls to library functions
; GCN-O2-NEXT:      Expand vector predication intrinsics
; GCN-O2-NEXT:      Scalarize Masked Memory Intrinsics
; GCN-O2-NEXT:      Expand reduction intrinsics
; GCN-O2-NEXT:      Early CSE
; GCN-O2-NEXT:    CallGraph Construction
; GCN-O2-NEXT:    Call Graph SCC Pass Manager
; GCN-O2-NEXT:      AMDGPU Annotate Kernel Features
; GCN-O2-NEXT:      FunctionPass Manager
; GCN-O2-NEXT:        AMDGPU Lower Kernel Arguments
; GCN-O2-NEXT:        Dominator Tree Construction
; GCN-O2-NEXT:        Natural Loop Information
; GCN-O2-NEXT:        CodeGen Prepare
; GCN-O2-NEXT:        Dominator Tree Construction
; GCN-O2-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:        Function Alias Analysis Results
; GCN-O2-NEXT:        Natural Loop Information
; GCN-O2-NEXT:        Scalar Evolution Analysis
; GCN-O2-NEXT:        GPU Load and Store Vectorizer
; GCN-O2-NEXT:        Lazy Value Information Analysis
; GCN-O2-NEXT:        Lower SwitchInst's to branches
; GCN-O2-NEXT:        Lower invoke and unwind, for unwindless code generators
; GCN-O2-NEXT:        Remove unreachable blocks from the CFG
; GCN-O2-NEXT:        Dominator Tree Construction
; GCN-O2-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:        Function Alias Analysis Results
; GCN-O2-NEXT:        Flatten the CFG
; GCN-O2-NEXT:        Dominator Tree Construction
; GCN-O2-NEXT:        Post-Dominator Tree Construction
; GCN-O2-NEXT:        Natural Loop Information
; GCN-O2-NEXT:        Legacy Divergence Analysis
; GCN-O2-NEXT:        AMDGPU IR late optimizations
; GCN-O2-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:        Function Alias Analysis Results
; GCN-O2-NEXT:        Code sinking
; GCN-O2-NEXT:        Legacy Divergence Analysis
; GCN-O2-NEXT:        Unify divergent function exit nodes
; GCN-O2-NEXT:        Lazy Value Information Analysis
; GCN-O2-NEXT:        Lower SwitchInst's to branches
; GCN-O2-NEXT:        Dominator Tree Construction
; GCN-O2-NEXT:        Natural Loop Information
; GCN-O2-NEXT:        Convert irreducible control-flow into natural loops
; GCN-O2-NEXT:        Fixup each natural loop to have a single exit block
; GCN-O2-NEXT:        Post-Dominator Tree Construction
; GCN-O2-NEXT:        Dominance Frontier Construction
; GCN-O2-NEXT:        Detect single entry single exit regions
; GCN-O2-NEXT:        Region Pass Manager
; GCN-O2-NEXT:          Structurize control flow
; GCN-O2-NEXT:        Post-Dominator Tree Construction
; GCN-O2-NEXT:        Natural Loop Information
; GCN-O2-NEXT:        Legacy Divergence Analysis
; GCN-O2-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:        Function Alias Analysis Results
; GCN-O2-NEXT:        Memory SSA
; GCN-O2-NEXT:        AMDGPU Annotate Uniform Values
; GCN-O2-NEXT:        SI annotate control flow
; GCN-O2-NEXT:        LCSSA Verifier
; GCN-O2-NEXT:        Loop-Closed SSA Form Pass
; GCN-O2-NEXT:      Analysis if a function is memory bound
; GCN-O2-NEXT:      DummyCGSCCPass
; GCN-O2-NEXT:      FunctionPass Manager
; GCN-O2-NEXT:        Safe Stack instrumentation pass
; GCN-O2-NEXT:        Insert stack protectors
; GCN-O2-NEXT:        Dominator Tree Construction
; GCN-O2-NEXT:        Post-Dominator Tree Construction
; GCN-O2-NEXT:        Natural Loop Information
; GCN-O2-NEXT:        Legacy Divergence Analysis
; GCN-O2-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O2-NEXT:        Function Alias Analysis Results
; GCN-O2-NEXT:        Branch Probability Analysis
; GCN-O2-NEXT:        Lazy Branch Probability Analysis
; GCN-O2-NEXT:        Lazy Block Frequency Analysis
; GCN-O2-NEXT:        AMDGPU DAG->DAG Pattern Instruction Selection
; GCN-O2-NEXT:        MachineDominator Tree Construction
; GCN-O2-NEXT:        SI Fix SGPR copies
; GCN-O2-NEXT:        MachinePostDominator Tree Construction
; GCN-O2-NEXT:        SI Lower i1 Copies
; GCN-O2-NEXT:        Finalize ISel and expand pseudo-instructions
; GCN-O2-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O2-NEXT:        Early Tail Duplication
; GCN-O2-NEXT:        Optimize machine instruction PHIs
; GCN-O2-NEXT:        Slot index numbering
; GCN-O2-NEXT:        Merge disjoint stack slots
; GCN-O2-NEXT:        Local Stack Slot Allocation
; GCN-O2-NEXT:        Remove dead machine instructions
; GCN-O2-NEXT:        MachineDominator Tree Construction
; GCN-O2-NEXT:        Machine Natural Loop Construction
; GCN-O2-NEXT:        Machine Block Frequency Analysis
; GCN-O2-NEXT:        Early Machine Loop Invariant Code Motion
; GCN-O2-NEXT:        MachineDominator Tree Construction
; GCN-O2-NEXT:        Machine Block Frequency Analysis
; GCN-O2-NEXT:        Machine Common Subexpression Elimination
; GCN-O2-NEXT:        MachinePostDominator Tree Construction
; GCN-O2-NEXT:        Machine code sinking
; GCN-O2-NEXT:        Peephole Optimizations
; GCN-O2-NEXT:        Remove dead machine instructions
; GCN-O2-NEXT:        SI Fold Operands
; GCN-O2-NEXT:        GCN DPP Combine
; GCN-O2-NEXT:        SI Load Store Optimizer
; GCN-O2-NEXT:        SI Peephole SDWA
; GCN-O2-NEXT:        Machine Block Frequency Analysis
; GCN-O2-NEXT:        MachineDominator Tree Construction
; GCN-O2-NEXT:        Early Machine Loop Invariant Code Motion
; GCN-O2-NEXT:        MachineDominator Tree Construction
; GCN-O2-NEXT:        Machine Block Frequency Analysis
; GCN-O2-NEXT:        Machine Common Subexpression Elimination
; GCN-O2-NEXT:        SI Fold Operands
; GCN-O2-NEXT:        Remove dead machine instructions
; GCN-O2-NEXT:        SI Shrink Instructions
; GCN-O2-NEXT:        Register Usage Information Propagation
; GCN-O2-NEXT:        Detect Dead Lanes
; GCN-O2-NEXT:        Remove dead machine instructions
; GCN-O2-NEXT:        Process Implicit Definitions
; GCN-O2-NEXT:        Remove unreachable machine basic blocks
; GCN-O2-NEXT:        Live Variable Analysis
; GCN-O2-NEXT:        SI Optimize VGPR LiveRange
; GCN-O2-NEXT:        Eliminate PHI nodes for register allocation
; GCN-O2-NEXT:        SI Lower control flow pseudo instructions
; GCN-O2-NEXT:        Two-Address instruction pass
; GCN-O2-NEXT:        Slot index numbering
; GCN-O2-NEXT:        Live Interval Analysis
; GCN-O2-NEXT:        Machine Natural Loop Construction
; GCN-O2-NEXT:        Simple Register Coalescing
; GCN-O2-NEXT:        Rename Disconnected Subregister Components
; GCN-O2-NEXT:        AMDGPU Pre-RA optimizations
; GCN-O2-NEXT:        Machine Instruction Scheduler
; GCN-O2-NEXT:        MachinePostDominator Tree Construction
; GCN-O2-NEXT:        SI Whole Quad Mode
; GCN-O2-NEXT:        Virtual Register Map
; GCN-O2-NEXT:        Live Register Matrix
; GCN-O2-NEXT:        SI Pre-allocate WWM Registers
; GCN-O2-NEXT:        SI optimize exec mask operations pre-RA
; GCN-O2-NEXT:        SI Form memory clauses
; GCN-O2-NEXT:        Machine Natural Loop Construction
; GCN-O2-NEXT:        Machine Block Frequency Analysis
; GCN-O2-NEXT:        Debug Variable Analysis
; GCN-O2-NEXT:        Live Stack Slot Analysis
; GCN-O2-NEXT:        Virtual Register Map
; GCN-O2-NEXT:        Live Register Matrix
; GCN-O2-NEXT:        Bundle Machine CFG Edges
; GCN-O2-NEXT:        Spill Code Placement Analysis
; GCN-O2-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O2-NEXT:        Machine Optimization Remark Emitter
; GCN-O2-NEXT:        Greedy Register Allocator
; GCN-O2-NEXT:        Virtual Register Rewriter
; GCN-O2-NEXT:        SI lower SGPR spill instructions
; GCN-O2-NEXT:        Virtual Register Map
; GCN-O2-NEXT:        Live Register Matrix
; GCN-O2-NEXT:        Greedy Register Allocator
; GCN-O2-NEXT:        GCN NSA Reassign
; GCN-O2-NEXT:        Virtual Register Rewriter
; GCN-O2-NEXT:        Stack Slot Coloring
; GCN-O2-NEXT:        Machine Copy Propagation Pass
; GCN-O2-NEXT:        Machine Loop Invariant Code Motion
; GCN-O2-NEXT:        SI Fix VGPR copies
; GCN-O2-NEXT:        SI optimize exec mask operations
; GCN-O2-NEXT:        Remove Redundant DEBUG_VALUE analysis
; GCN-O2-NEXT:        Fixup Statepoint Caller Saved
; GCN-O2-NEXT:        PostRA Machine Sink
; GCN-O2-NEXT:        MachineDominator Tree Construction
; GCN-O2-NEXT:        Machine Natural Loop Construction
; GCN-O2-NEXT:        Machine Block Frequency Analysis
; GCN-O2-NEXT:        MachinePostDominator Tree Construction
; GCN-O2-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O2-NEXT:        Machine Optimization Remark Emitter
; GCN-O2-NEXT:        Shrink Wrapping analysis
; GCN-O2-NEXT:        Prologue/Epilogue Insertion & Frame Finalization
; GCN-O2-NEXT:        Control Flow Optimizer
; GCN-O2-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O2-NEXT:        Tail Duplication
; GCN-O2-NEXT:        Machine Copy Propagation Pass
; GCN-O2-NEXT:        Post-RA pseudo instruction expansion pass
; GCN-O2-NEXT:        SI post-RA bundler
; GCN-O2-NEXT:        MachineDominator Tree Construction
; GCN-O2-NEXT:        Machine Natural Loop Construction
; GCN-O2-NEXT:        PostRA Machine Instruction Scheduler
; GCN-O2-NEXT:        Machine Block Frequency Analysis
; GCN-O2-NEXT:        MachinePostDominator Tree Construction
; GCN-O2-NEXT:        Branch Probability Basic Block Placement
; GCN-O2-NEXT:        Insert fentry calls
; GCN-O2-NEXT:        Insert XRay ops
; GCN-O2-NEXT:        SI Memory Legalizer
; GCN-O2-NEXT:        MachinePostDominator Tree Construction
; GCN-O2-NEXT:        SI insert wait instructions
; GCN-O2-NEXT:        SI Shrink Instructions
; GCN-O2-NEXT:        Insert required mode register values
; GCN-O2-NEXT:        SI Insert Hard Clauses
; GCN-O2-NEXT:        MachineDominator Tree Construction
; GCN-O2-NEXT:        SI Final Branch Preparation
; GCN-O2-NEXT:        SI peephole optimizations
; GCN-O2-NEXT:        Post RA hazard recognizer
; GCN-O2-NEXT:        Branch relaxation pass
; GCN-O2-NEXT:        Register Usage Information Collector Pass
; GCN-O2-NEXT:        Live DEBUG_VALUE analysis
; GCN-O2-NEXT:      Function register usage analysis
; GCN-O2-NEXT:      FunctionPass Manager
; GCN-O2-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O2-NEXT:        Machine Optimization Remark Emitter
; GCN-O2-NEXT:        AMDGPU Assembly Printer
; GCN-O2-NEXT:        Free MachineFunction
; GCN-O2-NEXT:Pass Arguments:  -domtree
; GCN-O2-NEXT:  FunctionPass Manager
; GCN-O2-NEXT:    Dominator Tree Construction

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
; GCN-O3-NEXT:  ModulePass Manager
; GCN-O3-NEXT:    Pre-ISel Intrinsic Lowering
; GCN-O3-NEXT:    AMDGPU Printf lowering
; GCN-O3-NEXT:      FunctionPass Manager
; GCN-O3-NEXT:        Dominator Tree Construction
; GCN-O3-NEXT:    Lower ctors and dtors for AMDGPU
; GCN-O3-NEXT:    Fix function bitcasts for AMDGPU
; GCN-O3-NEXT:    AMDGPU Lower Kernel Calls
; GCN-O3-NEXT:    FunctionPass Manager
; GCN-O3-NEXT:      Early propagate attributes from kernels to functions
; GCN-O3-NEXT:    AMDGPU Lower Intrinsics
; GCN-O3-NEXT:    AMDGPU Inline All Functions
; GCN-O3-NEXT:    CallGraph Construction
; GCN-O3-NEXT:    Call Graph SCC Pass Manager
; GCN-O3-NEXT:      Inliner for always_inline functions
; GCN-O3-NEXT:    A No-Op Barrier Pass
; GCN-O3-NEXT:    Lower OpenCL enqueued blocks
; GCN-O3-NEXT:    Lower uses of LDS variables from non-kernel functions
; GCN-O3-NEXT:    FunctionPass Manager
; GCN-O3-NEXT:      Infer address spaces
; GCN-O3-NEXT:      Expand Atomic instructions
; GCN-O3-NEXT:      AMDGPU Promote Alloca
; GCN-O3-NEXT:      Dominator Tree Construction
; GCN-O3-NEXT:      SROA
; GCN-O3-NEXT:      Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:      Function Alias Analysis Results
; GCN-O3-NEXT:      Memory SSA
; GCN-O3-NEXT:      Natural Loop Information
; GCN-O3-NEXT:      Canonicalize natural loops
; GCN-O3-NEXT:      LCSSA Verifier
; GCN-O3-NEXT:      Loop-Closed SSA Form Pass
; GCN-O3-NEXT:      Scalar Evolution Analysis
; GCN-O3-NEXT:      Lazy Branch Probability Analysis
; GCN-O3-NEXT:      Lazy Block Frequency Analysis
; GCN-O3-NEXT:      Loop Pass Manager
; GCN-O3-NEXT:        Loop Invariant Code Motion
; GCN-O3-NEXT:      Split GEPs to a variadic base and a constant offset for better CSE
; GCN-O3-NEXT:      Speculatively execute instructions
; GCN-O3-NEXT:      Scalar Evolution Analysis
; GCN-O3-NEXT:      Straight line strength reduction
; GCN-O3-NEXT:      Phi Values Analysis
; GCN-O3-NEXT:      Function Alias Analysis Results
; GCN-O3-NEXT:      Memory Dependence Analysis
; GCN-O3-NEXT:      Optimization Remark Emitter
; GCN-O3-NEXT:      Global Value Numbering
; GCN-O3-NEXT:      Scalar Evolution Analysis
; GCN-O3-NEXT:      Nary reassociation
; GCN-O3-NEXT:      Early CSE
; GCN-O3-NEXT:      Post-Dominator Tree Construction
; GCN-O3-NEXT:      Legacy Divergence Analysis
; GCN-O3-NEXT:      AMDGPU IR optimizations
; GCN-O3-NEXT:      Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:      Canonicalize natural loops
; GCN-O3-NEXT:      Scalar Evolution Analysis
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
; GCN-O3-NEXT:      Lower constant intrinsics
; GCN-O3-NEXT:      Remove unreachable blocks from the CFG
; GCN-O3-NEXT:      Natural Loop Information
; GCN-O3-NEXT:      Post-Dominator Tree Construction
; GCN-O3-NEXT:      Branch Probability Analysis
; GCN-O3-NEXT:      Block Frequency Analysis
; GCN-O3-NEXT:      Constant Hoisting
; GCN-O3-NEXT:      Replace intrinsics with calls to vector library
; GCN-O3-NEXT:      Partially inline calls to library functions
; GCN-O3-NEXT:      Expand vector predication intrinsics
; GCN-O3-NEXT:      Scalarize Masked Memory Intrinsics
; GCN-O3-NEXT:      Expand reduction intrinsics
; GCN-O3-NEXT:      Natural Loop Information
; GCN-O3-NEXT:      Phi Values Analysis
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
; GCN-O3-NEXT:        Post-Dominator Tree Construction
; GCN-O3-NEXT:        Natural Loop Information
; GCN-O3-NEXT:        Legacy Divergence Analysis
; GCN-O3-NEXT:        AMDGPU IR late optimizations
; GCN-O3-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:        Function Alias Analysis Results
; GCN-O3-NEXT:        Code sinking
; GCN-O3-NEXT:        Legacy Divergence Analysis
; GCN-O3-NEXT:        Unify divergent function exit nodes
; GCN-O3-NEXT:        Lazy Value Information Analysis
; GCN-O3-NEXT:        Lower SwitchInst's to branches
; GCN-O3-NEXT:        Dominator Tree Construction
; GCN-O3-NEXT:        Natural Loop Information
; GCN-O3-NEXT:        Convert irreducible control-flow into natural loops
; GCN-O3-NEXT:        Fixup each natural loop to have a single exit block
; GCN-O3-NEXT:        Post-Dominator Tree Construction
; GCN-O3-NEXT:        Dominance Frontier Construction
; GCN-O3-NEXT:        Detect single entry single exit regions
; GCN-O3-NEXT:        Region Pass Manager
; GCN-O3-NEXT:          Structurize control flow
; GCN-O3-NEXT:        Post-Dominator Tree Construction
; GCN-O3-NEXT:        Natural Loop Information
; GCN-O3-NEXT:        Legacy Divergence Analysis
; GCN-O3-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:        Function Alias Analysis Results
; GCN-O3-NEXT:        Memory SSA
; GCN-O3-NEXT:        AMDGPU Annotate Uniform Values
; GCN-O3-NEXT:        SI annotate control flow
; GCN-O3-NEXT:        LCSSA Verifier
; GCN-O3-NEXT:        Loop-Closed SSA Form Pass
; GCN-O3-NEXT:      Analysis if a function is memory bound
; GCN-O3-NEXT:      DummyCGSCCPass
; GCN-O3-NEXT:      FunctionPass Manager
; GCN-O3-NEXT:        Safe Stack instrumentation pass
; GCN-O3-NEXT:        Insert stack protectors
; GCN-O3-NEXT:        Dominator Tree Construction
; GCN-O3-NEXT:        Post-Dominator Tree Construction
; GCN-O3-NEXT:        Natural Loop Information
; GCN-O3-NEXT:        Legacy Divergence Analysis
; GCN-O3-NEXT:        Basic Alias Analysis (stateless AA impl)
; GCN-O3-NEXT:        Function Alias Analysis Results
; GCN-O3-NEXT:        Branch Probability Analysis
; GCN-O3-NEXT:        Lazy Branch Probability Analysis
; GCN-O3-NEXT:        Lazy Block Frequency Analysis
; GCN-O3-NEXT:        AMDGPU DAG->DAG Pattern Instruction Selection
; GCN-O3-NEXT:        MachineDominator Tree Construction
; GCN-O3-NEXT:        SI Fix SGPR copies
; GCN-O3-NEXT:        MachinePostDominator Tree Construction
; GCN-O3-NEXT:        SI Lower i1 Copies
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
; GCN-O3-NEXT:        Process Implicit Definitions
; GCN-O3-NEXT:        Remove unreachable machine basic blocks
; GCN-O3-NEXT:        Live Variable Analysis
; GCN-O3-NEXT:        SI Optimize VGPR LiveRange
; GCN-O3-NEXT:        Eliminate PHI nodes for register allocation
; GCN-O3-NEXT:        SI Lower control flow pseudo instructions
; GCN-O3-NEXT:        Two-Address instruction pass
; GCN-O3-NEXT:        Slot index numbering
; GCN-O3-NEXT:        Live Interval Analysis
; GCN-O3-NEXT:        Machine Natural Loop Construction
; GCN-O3-NEXT:        Simple Register Coalescing
; GCN-O3-NEXT:        Rename Disconnected Subregister Components
; GCN-O3-NEXT:        AMDGPU Pre-RA optimizations
; GCN-O3-NEXT:        Machine Instruction Scheduler
; GCN-O3-NEXT:        MachinePostDominator Tree Construction
; GCN-O3-NEXT:        SI Whole Quad Mode
; GCN-O3-NEXT:        Virtual Register Map
; GCN-O3-NEXT:        Live Register Matrix
; GCN-O3-NEXT:        SI Pre-allocate WWM Registers
; GCN-O3-NEXT:        SI optimize exec mask operations pre-RA
; GCN-O3-NEXT:        SI Form memory clauses
; GCN-O3-NEXT:        Machine Natural Loop Construction
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
; GCN-O3-NEXT:        SI lower SGPR spill instructions
; GCN-O3-NEXT:        Virtual Register Map
; GCN-O3-NEXT:        Live Register Matrix
; GCN-O3-NEXT:        Greedy Register Allocator
; GCN-O3-NEXT:        GCN NSA Reassign
; GCN-O3-NEXT:        Virtual Register Rewriter
; GCN-O3-NEXT:        Stack Slot Coloring
; GCN-O3-NEXT:        Machine Copy Propagation Pass
; GCN-O3-NEXT:        Machine Loop Invariant Code Motion
; GCN-O3-NEXT:        SI Fix VGPR copies
; GCN-O3-NEXT:        SI optimize exec mask operations
; GCN-O3-NEXT:        Remove Redundant DEBUG_VALUE analysis
; GCN-O3-NEXT:        Fixup Statepoint Caller Saved
; GCN-O3-NEXT:        PostRA Machine Sink
; GCN-O3-NEXT:        MachineDominator Tree Construction
; GCN-O3-NEXT:        Machine Natural Loop Construction
; GCN-O3-NEXT:        Machine Block Frequency Analysis
; GCN-O3-NEXT:        MachinePostDominator Tree Construction
; GCN-O3-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O3-NEXT:        Machine Optimization Remark Emitter
; GCN-O3-NEXT:        Shrink Wrapping analysis
; GCN-O3-NEXT:        Prologue/Epilogue Insertion & Frame Finalization
; GCN-O3-NEXT:        Control Flow Optimizer
; GCN-O3-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O3-NEXT:        Tail Duplication
; GCN-O3-NEXT:        Machine Copy Propagation Pass
; GCN-O3-NEXT:        Post-RA pseudo instruction expansion pass
; GCN-O3-NEXT:        SI post-RA bundler
; GCN-O3-NEXT:        MachineDominator Tree Construction
; GCN-O3-NEXT:        Machine Natural Loop Construction
; GCN-O3-NEXT:        PostRA Machine Instruction Scheduler
; GCN-O3-NEXT:        Machine Block Frequency Analysis
; GCN-O3-NEXT:        MachinePostDominator Tree Construction
; GCN-O3-NEXT:        Branch Probability Basic Block Placement
; GCN-O3-NEXT:        Insert fentry calls
; GCN-O3-NEXT:        Insert XRay ops
; GCN-O3-NEXT:        SI Memory Legalizer
; GCN-O3-NEXT:        MachinePostDominator Tree Construction
; GCN-O3-NEXT:        SI insert wait instructions
; GCN-O3-NEXT:        SI Shrink Instructions
; GCN-O3-NEXT:        Insert required mode register values
; GCN-O3-NEXT:        SI Insert Hard Clauses
; GCN-O3-NEXT:        MachineDominator Tree Construction
; GCN-O3-NEXT:        SI Final Branch Preparation
; GCN-O3-NEXT:        SI peephole optimizations
; GCN-O3-NEXT:        Post RA hazard recognizer
; GCN-O3-NEXT:        Branch relaxation pass
; GCN-O3-NEXT:        Register Usage Information Collector Pass
; GCN-O3-NEXT:        Live DEBUG_VALUE analysis
; GCN-O3-NEXT:      Function register usage analysis
; GCN-O3-NEXT:      FunctionPass Manager
; GCN-O3-NEXT:        Lazy Machine Block Frequency Analysis
; GCN-O3-NEXT:        Machine Optimization Remark Emitter
; GCN-O3-NEXT:        AMDGPU Assembly Printer
; GCN-O3-NEXT:        Free MachineFunction
; GCN-O3-NEXT:Pass Arguments:  -domtree
; GCN-O3-NEXT:  FunctionPass Manager
; GCN-O3-NEXT:    Dominator Tree Construction

define void @empty() {
  ret void
}
