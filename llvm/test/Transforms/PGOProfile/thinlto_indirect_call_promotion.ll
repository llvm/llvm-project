; The raw profiles and reduced IR inputs are generated from Inputs/update_icall_promotion_inputs.sh

; Do setup work for all below tests: annotate value profiles, generate bitcode and combined index.
; Explicitly turn off ICP pass in Inputs/thinlto_indirect_call_promotion.ll.
; This way ICP happens in %t.bc after _Z11global_funcv and two indirect callees are imported.
; RUN: opt -passes=pgo-instr-use -pgo-test-profile-file=%p/Inputs/thinlto_icall_prom.profdata -module-summary %s -o %t.bc
; RUN: opt -disable-icp -passes=pgo-instr-use -pgo-test-profile-file=%p/Inputs/thinlto_icall_prom.profdata -module-summary %p/Inputs/thinlto_indirect_call_promotion.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Test that callee with local linkage has `PGOFuncName` metadata while callee with external doesn't have it.
; RUN: llvm-dis %t2.bc -o - | FileCheck %s --check-prefix=PGOName
; PGOName: define internal void @_ZL7callee0v() {{.*}} !prof !{{[0-9]+}} !PGOFuncName ![[MD:[0-9]+]] {
; PGOName: define void @_Z7callee1v() {{.*}} !prof !{{[0-9]+}} {
; PGOName: ![[MD]] = !{!"lib.cc;_ZL7callee0v"}

; Tests that both external and internal callees are correctly imported.
; RUN: opt -passes=function-import -summary-file %t3.thinlto.bc %t.bc -o %t4.bc -print-imports 2>&1 | FileCheck %s --check-prefix=IMPORTS
; IMPORTS: Import _ZL7callee0v.llvm{{.*}}
; IMPORTS: Import _Z7callee1v
; IMPORTS: Import _Z11global_funcv

; Tests that ICP transformations happen.
; Both candidates are ICP'ed, check there is no `!VP` in the IR.
; RUN: opt %t4.bc -icp-lto -passes=pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-PROM --implicit-check-not="!VP"
; RUN: opt %t4.bc -icp-lto -passes=pgo-icall-prom -S -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefix=PASS-REMARK

; PASS-REMARK: Promote indirect call to _ZL7callee0v.llvm.0 with count 3 out of 5
; PASS-REMARK: Promote indirect call to _Z7callee1v with count 2 out of 2

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  tail call void @_Z11global_funcv()
  ret i32 0
}

declare void @_Z11global_funcv()

; ICALL-PROM:   br i1 %{{[0-9]+}}, label %if.true.direct_targ, label %if.false.orig_indirect, !prof [[BRANCH_WEIGHT1:![0-9]+]]
; ICALL-PROM:   br i1 %{{[0-9]+}}, label %if.true.direct_targ1, label %if.false.orig_indirect2, !prof [[BRANCH_WEIGHT2:![0-9]+]]

; ICALL-PROM: [[BRANCH_WEIGHT1]] = !{!"branch_weights", i32 3, i32 2}
; ICALL-PROM: [[BRANCH_WEIGHT2]] = !{!"branch_weights", i32 2, i32 0}
