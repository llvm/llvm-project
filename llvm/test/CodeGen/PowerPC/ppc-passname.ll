; Test pass name: ppc-loop-instr-form-prep.
; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-before=ppc-loop-instr-form-prep -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-BEFORE-LOOP-INSTR-FORM-PREP
; STOP-BEFORE-LOOP-INSTR-FORM-PREP-NOT: -ppc-loop-instr-form-prep
; STOP-BEFORE-LOOP-INSTR-FORM-PREP-NOT: "ppc-loop-instr-form-prep" pass is not registered.
; STOP-BEFORE-LOOP-INSTR-FORM-PREP-NOT: Prepare loop for ppc preferred instruction forms

; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-after=ppc-loop-instr-form-prep -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-AFTER-LOOP-INSTR-FORM-PREP
; STOP-AFTER-LOOP-INSTR-FORM-PREP: -ppc-loop-instr-form-prep
; STOP-AFTER-LOOP-INSTR-FORM-PREP-NOT: "ppc-loop-instr-form-prep" pass is not registered.
; STOP-AFTER-LOOP-INSTR-FORM-PREP: Prepare loop for ppc preferred instruction forms


; Test pass name: ppc-toc-reg-deps.
; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-before=ppc-toc-reg-deps -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-BEFORE-TOC-REG-DEPS
; STOP-BEFORE-TOC-REG-DEPS-NOT: -ppc-toc-reg-deps
; STOP-BEFORE-TOC-REG-DEPS-NOT: "ppc-toc-reg-deps" pass is not registered.
; STOP-BEFORE-TOC-REG-DEPS-NOT: PowerPC TOC Register Dependencies

; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-after=ppc-toc-reg-deps -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-AFTER-TOC-REG-DEPS
; STOP-AFTER-TOC-REG-DEPS: -ppc-toc-reg-deps
; STOP-AFTER-TOC-REG-DEPS-NOT: "ppc-toc-reg-deps" pass is not registered.
; STOP-AFTER-TOC-REG-DEPS: PowerPC TOC Register Dependencies


; Test pass name: ppc-vsx-copy.
; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-before=ppc-vsx-copy -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-BEFORE-VSX-COPY
; STOP-BEFORE-VSX-COPY-NOT: -ppc-vsx-copy
; STOP-BEFORE-VSX-COPY-NOT: "ppc-vsx-copy" pass is not registered.
; STOP-BEFORE-VSX-COPY-NOT: PowerPC VSX Copy Legalization

; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-after=ppc-vsx-copy -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-AFTER-VSX-COPY
; STOP-AFTER-VSX-COPY: -ppc-vsx-copy
; STOP-AFTER-VSX-COPY-NOT: "ppc-vsx-copy" pass is not registered.
; STOP-AFTER-VSX-COPY: PowerPC VSX Copy Legalization


; Test pass name: ppc-early-ret.
; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-before=ppc-early-ret -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-BEFORE-EARLY-RET
; STOP-BEFORE-EARLY-RET-NOT: -ppc-early-ret
; STOP-BEFORE-EARLY-RET-NOT: "ppc-early-ret" pass is not registered.
; STOP-BEFORE-EARLY-RET-NOT: PowerPC Early-Return Creation

; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-after=ppc-early-ret -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-AFTER-EARLY-RET
; STOP-AFTER-EARLY-RET: -ppc-early-ret
; STOP-AFTER-EARLY-RET-NOT: "ppc-early-ret" pass is not registered.
; STOP-AFTER-EARLY-RET: PowerPC Early-Return Creation


; Test pass name: ppc-vsx-fma-mutate.
; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-before=ppc-vsx-fma-mutate -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-BEFORE-VSX-FMA-MUTATE
; STOP-BEFORE-VSX-FMA-MUTATE-NOT: -ppc-vsx-fma-mutate
; STOP-BEFORE-VSX-FMA-MUTATE-NOT: "ppc-vsx-fma-mutate" pass is not registered.
; STOP-BEFORE-VSX-FMA-MUTATE-NOT: PowerPC VSX FMA Mutation

; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-after=ppc-vsx-fma-mutate -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-AFTER-VSX-FMA-MUTATE
; STOP-AFTER-VSX-FMA-MUTATE: -ppc-vsx-fma-mutate
; STOP-AFTER-VSX-FMA-MUTATE-NOT: "ppc-vsx-fma-mutate" pass is not registered.
; STOP-AFTER-VSX-FMA-MUTATE: PowerPC VSX FMA Mutation


; Test pass name: ppc-vsx-swaps.
; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-before=ppc-vsx-swaps -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-BEFORE-VSX-SWAPS
; STOP-BEFORE-VSX-SWAPS-NOT: -ppc-vsx-swaps
; STOP-BEFORE-VSX-SWAPS-NOT: "ppc-vsx-swaps" pass is not registered.
; STOP-BEFORE-VSX-SWAPS-NOT: PowerPC VSX Swap Removal

; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-after=ppc-vsx-swaps -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-AFTER-VSX-SWAPS
; STOP-AFTER-VSX-SWAPS: -ppc-vsx-swaps
; STOP-AFTER-VSX-SWAPS-NOT: "ppc-vsx-swaps" pass is not registered.
; STOP-AFTER-VSX-SWAPS: PowerPC VSX Swap Removal


; Test pass name: ppc-reduce-cr-ops.
; RUN: llc -ppc-reduce-cr-logicals -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-before=ppc-reduce-cr-ops -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-BEFORE-REDUCE-CR-OPS
; STOP-BEFORE-REDUCE-CR-OPS-NOT: -ppc-reduce-cr-ops
; STOP-BEFORE-REDUCE-CR-OPS-NOT: "ppc-reduce-cr-ops" pass is not registered.
; STOP-BEFORE-REDUCE-CR-OPS-NOT: PowerPC Reduce CR logical Operation

; RUN: llc -ppc-reduce-cr-logicals -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-after=ppc-reduce-cr-ops -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-AFTER-REDUCE-CR-OPS
; STOP-AFTER-REDUCE-CR-OPS: -ppc-reduce-cr-ops
; STOP-AFTER-REDUCE-CR-OPS-NOT: "ppc-reduce-cr-ops" pass is not registered.
; STOP-AFTER-REDUCE-CR-OPS: PowerPC Reduce CR logical Operation


; Test pass name: ppc-branch-select.
; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-before=ppc-branch-select -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-BEFORE-BRANCH-SELECT
; STOP-BEFORE-BRANCH-SELECT-NOT: -ppc-branch-select
; STOP-BEFORE-BRANCH-SELECT-NOT: "ppc-branch-select" pass is not registered.
; STOP-BEFORE-BRANCH-SELECT-NOT: PowerPC Branch Selector

; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-after=ppc-branch-select -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-AFTER-BRANCH-SELECT
; STOP-AFTER-BRANCH-SELECT: -ppc-branch-select
; STOP-AFTER-BRANCH-SELECT-NOT: "ppc-branch-select" pass is not registered.
; STOP-AFTER-BRANCH-SELECT: PowerPC Branch Selector


; Test pass name: ppc-branch-coalescing.
; RUN: llc -enable-ppc-branch-coalesce -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-before=ppc-branch-coalescing -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-BEFORE-BRANCH-COALESCING
; STOP-BEFORE-BRANCH-COALESCING-NOT: -ppc-branch-coalescing
; STOP-BEFORE-BRANCH-COALESCING-NOT: "ppc-branch-coalescing" pass is not registered.
; STOP-BEFORE-BRANCH-COALESCING-NOT: Branch Coalescing 

; RUN: llc -enable-ppc-branch-coalesce -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-after=ppc-branch-coalescing -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-AFTER-BRANCH-COALESCING
; STOP-AFTER-BRANCH-COALESCING: -ppc-branch-coalescing
; STOP-AFTER-BRANCH-COALESCING-NOT: "ppc-branch-coalescing" pass is not registered.
; STOP-AFTER-BRANCH-COALESCING: Branch Coalescing 

