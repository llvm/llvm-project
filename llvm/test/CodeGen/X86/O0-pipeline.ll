; When EXPENSIVE_CHECKS are enabled, the machine verifier appears between each
; pass. Ignore it with 'grep -v'.
; RUN: llc -mtriple=x86_64-- -O0 -debug-pass=Structure < %s -o /dev/null 2>&1 \
; RUN:   | grep -v 'Verify generated machine code' | FileCheck %s  --check-prefixes=COMMON,SDISEL,DEFAULT
; RUN: llc -mtriple=x86_64-- -O0 -fast-isel=0 -debug-pass=Structure < %s -o /dev/null 2>&1 \
; RUN:   | grep -v 'Verify generated machine code' | FileCheck %s  --check-prefixes=COMMON,SDISEL,NOFASTISEL
; RUN: llc -mtriple=x86_64-- -O0 -global-isel -fast-isel=0 -debug-pass=Structure < %s -o /dev/null 2>&1 \
; RUN:   | grep -v 'Verify generated machine code' | FileCheck %s  --check-prefixes=COMMON,GISEL,NOFASTISEL

; REQUIRES: asserts

; COMMON-LABEL: Pass Arguments:
; COMMON-NEXT: Target Library Information
; COMMON-NEXT: Target Pass Configuration
; COMMON-NEXT: Machine Module Information
; COMMON-NEXT: Target Transform Information
; COMMON-NEXT: Create Garbage Collector Module Metadata
; COMMON-NEXT: Assumption Cache Tracker
; SDISEL-NEXT: Profile summary info
; COMMON-NEXT: Machine Branch Probability Analysis
; COMMON-NEXT:   ModulePass Manager
; COMMON-NEXT:     Pre-ISel Intrinsic Lowering
; COMMON-NEXT:     FunctionPass Manager
; COMMON-NEXT:       Expand large div/rem
; COMMON-NEXT:       Expand large fp convert
; COMMON-NEXT:       Expand Atomic instructions
; COMMON-NEXT:       Lower AMX intrinsics
; COMMON-NEXT:       Lower AMX type for load/store
; COMMON-NEXT:       Module Verifier
; COMMON-NEXT:       Lower Garbage Collection Instructions
; COMMON-NEXT:       Shadow Stack GC Lowering
; COMMON-NEXT:       Lower constant intrinsics
; COMMON-NEXT:       Remove unreachable blocks from the CFG
; COMMON-NEXT:       Expand vector predication intrinsics
; COMMON-NEXT:       Scalarize Masked Memory Intrinsics
; COMMON-NEXT:       Expand reduction intrinsics
; COMMON-NEXT:       Expand indirectbr instructions
; COMMON-NEXT:       Exception handling preparation
; COMMON-NEXT:       Prepare callbr
; COMMON-NEXT:       Safe Stack instrumentation pass
; COMMON-NEXT:       Insert stack protectors
; COMMON-NEXT:       Module Verifier
; SDISEL-NEXT:       Assignment Tracking Analysis
; SDISEL-NEXT:       X86 DAG->DAG Instruction Selection
; SDISEL-NEXT:       X86 PIC Global Base Reg Initialization
; SDISEL-NEXT:       Argument Stack Rebase
; GISEL-NEXT:        Analysis containing CSE Info
; GISEL-NEXT:        IRTranslator
; GISEL-NEXT:        Analysis containing CSE Info
; GISEL-NEXT:        Analysis for ComputingKnownBits
; GISEL-NEXT:        Legalizer
; GISEL-NEXT:        RegBankSelect
; GISEL-NEXT:        Analysis for ComputingKnownBits
; GISEL-NEXT:        InstructionSelect
; GISEL-NEXT:        ResetMachineFunction
; COMMON-NEXT:       Finalize ISel and expand pseudo-instructions
; COMMON-NEXT:       Local Stack Slot Allocation
; COMMON-NEXT:       X86 speculative load hardening
; COMMON-NEXT:       MachineDominator Tree Construction
; COMMON-NEXT:       X86 EFLAGS copy lowering
; COMMON-NEXT:       X86 DynAlloca Expander
; DEFAULT-NEXT:      Fast Tile Register Preconfigure
; NOFASTISEL-NEXT:   MachineDominator Tree Construction
; NOFASTISEL-NEXT:   Machine Natural Loop Construction
; NOFASTISEL-NEXT:   Tile Register Pre-configure
; COMMON-NEXT:       Eliminate PHI nodes for register allocation
; COMMON-NEXT:       Two-Address instruction pass
; COMMON-NEXT:       Fast Register Allocator
; COMMON-NEXT:       Fast Tile Register Configure
; COMMON-NEXT:       X86 Lower Tile Copy
; COMMON-NEXT:       Bundle Machine CFG Edges
; COMMON-NEXT:       X86 FP Stackifier
; COMMON-NEXT:       Remove Redundant DEBUG_VALUE analysis
; COMMON-NEXT:       Fixup Statepoint Caller Saved
; COMMON-NEXT:       Lazy Machine Block Frequency Analysis
; COMMON-NEXT:       Machine Optimization Remark Emitter
; COMMON-NEXT:       Prologue/Epilogue Insertion & Frame Finalization
; COMMON-NEXT:       Post-RA pseudo instruction expansion pass
; COMMON-NEXT:       X86 pseudo instruction expansion pass
; COMMON-NEXT:       Insert KCFI indirect call checks
; COMMON-NEXT:       Analyze Machine Code For Garbage Collection
; COMMON-NEXT:       Insert fentry calls
; COMMON-NEXT:       Insert XRay ops
; COMMON-NEXT:       Implement the 'patchable-function' attribute
; COMMON-NEXT:       X86 Indirect Branch Tracking
; COMMON-NEXT:       X86 vzeroupper inserter
; COMMON-NEXT:       Compressing EVEX instrs to VEX encoding when possibl
; COMMON-NEXT:       X86 Discriminate Memory Operands
; COMMON-NEXT:       X86 Insert Cache Prefetches
; COMMON-NEXT:       X86 insert wait instruction
; COMMON-NEXT:       Contiguously Lay Out Funclets
; COMMON-NEXT:       StackMap Liveness Analysis
; COMMON-NEXT:       Live DEBUG_VALUE analysis
; COMMON-NEXT:       Machine Sanitizer Binary Metadata
; COMMON-NEXT:       Lazy Machine Block Frequency Analysis
; COMMON-NEXT:       Machine Optimization Remark Emitter
; COMMON-NEXT:       Stack Frame Layout Analysis
; COMMON-NEXT:       X86 Speculative Execution Side Effect Suppression
; COMMON-NEXT:       X86 Indirect Thunks
; COMMON-NEXT:       X86 Return Thunks
; COMMON-NEXT:       Check CFA info and insert CFI instructions if needed
; COMMON-NEXT:       X86 Load Value Injection (LVI) Ret-Hardening
; COMMON-NEXT:       Pseudo Probe Inserter
; COMMON-NEXT:       Unpack machine instruction bundles
; COMMON-NEXT:       Lazy Machine Block Frequency Analysis
; COMMON-NEXT:       Machine Optimization Remark Emitter
; COMMON-NEXT:       X86 Assembly Printer
; COMMON-NEXT:       Free MachineFunction

define void @f() {
  ret void
}
