; RUN: opt < %s -o /dev/null -enable-new-pm=0 -block-freq -opt-remark-emitter -memoryssa -inject-tli-mappings -pgo-memop-opt -verify-loop-info -debug-pass=Details 2>&1 | FileCheck %s

; REQUIRES: asserts

; This is a heavily reduced reproducer for the problem found in
; https://bugs.llvm.org/show_bug.cgi?id=49950 when doing fuzzy
; testing (including non-standard pipelines).
;
; The problem manifested as having a pass structure like this
; when it failed (as given by using -debug-pass=Details):
;
;   Target Library Information
;   Target Transform Information
;   Profile summary info
;   Assumption Cache Tracker
;     ModulePass Manager
;       FunctionPass Manager
;         Dominator Tree Construction
;         Natural Loop Information
;         Post-Dominator Tree Construction
;         Branch Probability Analysis
;         Block Frequency Analysis
;   --      Branch Probability Analysis
;         Lazy Branch Probability Analysis
;         Lazy Block Frequency Analysis
;         Optimization Remark Emitter
;         Basic Alias Analysis (stateless AA impl)
;         Function Alias Analysis Results
;         Memory SSA
;   --      Dominator Tree Construction
;   --      Function Alias Analysis Results
;   --      Basic Alias Analysis (stateless AA impl)
;   --      Memory SSA
;         Inject TLI Mappings
;   --      Inject TLI Mappings
;         PGOMemOPSize
;   --      Block Frequency Analysis
;   --      Post-Dominator Tree Construction
;   --      Optimization Remark Emitter
;   --      Lazy Branch Probability Analysis
;   --      Natural Loop Information
;   --      Lazy Block Frequency Analysis
;   --      PGOMemOPSize
;         Module Verifier
;   --      Module Verifier
;   --    Target Library Information
;   --    Profile summary info
;   --    Assumption Cache Tracker
;       Bitcode Writer
;   --    Bitcode Writer
;
; One might notice that "Dominator Tree Construction" is dropped after
; "Memory SSA", while for example "Natural Loop Information" stick around
; a bit longer. This despite "Dominator Tree Construction" being transitively
; required by "Natural Loop Information".
; The end result was that we got crashes when doing verification of loop
; info after "Inject TLI Mappings" (since the dominator tree had been
; removed too early).

; Verify that both domintator tree and loop info are kept until after
; PGOMemOPSize:
;
; CHECK:     Dominator Tree Construction
; CHECK-NOT: --      Dominator Tree Construction
; CHECK:     Memory SSA
; CHECK-NOT: --      Dominator Tree Construction
; CHECK:     Inject TLI Mappings
; CHECK-NOT: --      Dominator Tree Construction
; CHECK:     PGOMemOPSize
; CHECK-DAG: --      Dominator Tree Construction
; CHECK-DAG: --      Natural Loop Information
; CHECK-DAG: --      PGOMemOPSize
; CHECK:     Bitcode Writer

define void @foo() {
entry:
  ret void
}