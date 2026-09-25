; RUN: llc -enable-new-pm -mtriple=riscv32 -O0 -print-pipeline-passes=tree < %s 2>&1 \
; RUN:   | grep -v verify \
; RUN:   | FileCheck %s
; RUN: llc -enable-new-pm -mtriple=riscv64 -O0 -print-pipeline-passes=tree < %s 2>&1 \
; RUN:   | grep -v verify \
; RUN:   | FileCheck %s

; REQUIRES: asserts

; CHECK: require<MachineModuleAnalysis>
; CHECK-NEXT: require<profile-summary>
; CHECK-NEXT: require<collector-metadata>
; CHECK-NEXT: require<runtime-libcall-info>
; CHECK-NEXT: require<libcall-lowering-info>
; CHECK-NEXT: pre-isel-intrinsic-lowering
; CHECK-NEXT: function
; CHECK-NEXT:   expand-ir-insts<O0>
; CHECK-NEXT:   atomic-expand
; CHECK-NEXT:   riscv-zacas-abi-fix
; CHECK-NEXT:   gc-lowering
; CHECK-NEXT: shadow-stack-gc-lowering
; CHECK-NEXT: function
; CHECK-NEXT:   unreachableblockelim
; CHECK-NEXT:   ee-instrument<post-inline>
; CHECK-NEXT:   scalarize-masked-mem-intrin
; CHECK-NEXT:   expand-reductions
; CHECK-NEXT:   dwarf-eh-prepare
; CHECK-NEXT:   inline-asm-prepare
; CHECK-NEXT:   safe-stack
; CHECK-NEXT:   stack-protector
; CHECK-NEXT: riscv-asm-printer-begin
; CHECK-NEXT: function
; CHECK-NEXT:   machine-function
; CHECK-NEXT:     riscv-isel
; CHECK-NEXT:     finalize-isel
; CHECK-NEXT:     localstackalloc
; CHECK-NEXT:     riscv-expand-pseudo-pre-ra
; CHECK-NEXT:     phi-node-elimination
; CHECK-NEXT:     two-address-instruction
; CHECK-NEXT:     regallocfast
; CHECK-NEXT:     remove-redundant-debug-values
; CHECK-NEXT:     fixup-statepoint-caller-saved
; CHECK-NEXT:     prolog-epilog
; CHECK-NEXT:     post-ra-pseudos
; CHECK-NEXT:     riscv-expand-pseudo-post-ra
; CHECK-NEXT:     kcfi
; CHECK-NEXT:     fentry-insert
; CHECK-NEXT:     xray-instrumentation
; CHECK-NEXT:     patchable-function
; CHECK-NEXT:     branch-relaxation
; CHECK-NEXT:     funclet-layout
; CHECK-NEXT:     remove-loads-into-fake-uses
; CHECK-NEXT:     StackMapLivenessPass
; CHECK-NEXT:     live-debug-values<emit-debug-entry-values>
; CHECK-NEXT:     machine-sanmd
; CHECK-NEXT:     stack-frame-layout
; CHECK-NEXT:     riscv-expand-pseudo-pre-emit
; CHECK-NEXT:     riscv-expand-pseudo-atomics
; CHECK-NEXT:     unpack-mi-bundles
; CHECK-NEXT:     riscv-asm-printer
; CHECK-NEXT:   free-machine-function
; CHECK-NEXT: riscv-asm-printer-end
