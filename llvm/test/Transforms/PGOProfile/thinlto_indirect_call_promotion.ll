; The raw profiles (and reduced IR if needed) could be re-generated (e.g., when
; there is a profile version bump) from script
; Inputs/update_thinlto_indirect_call_promotion_inputs.sh
;
; The script generates raw profiles. This regression test will convert it to
; indexed profiles. This way the test exercises code path where a profiled
; callee address in raw profiles is converted to function hash in index profiles.

; The raw profiles storesd compressed function names, so profile reader should
; be built with zlib support to decompress them.
; REQUIRES: zlib

; RUN: rm -rf %t && split-file %s %t && cd %t

; Do setup work for all below tests: convert raw profiles to indexed profiles,
; run profile-use pass, generate bitcode and combined ThinLTO index.
; Note `pgo-instr-use` pass runs without `pgo-icall-prom` pass. As a result ICP
; transformation won't happen at test setup time.
; RUN: llvm-profdata merge %p/Inputs/thinlto_indirect_call_promotion.profraw -o icp.profdata
; RUN: opt -passes=pgo-instr-use -pgo-test-profile-file=icp.profdata -module-summary main.ll -o main.bc
; RUN: opt -passes=pgo-instr-use -pgo-test-profile-file=icp.profdata -module-summary lib.ll -o lib.bc
; RUN: llvm-lto -thinlto -o summary main.bc lib.bc

; Test that callee with local linkage has `PGOFuncName` metadata while callee with external doesn't have it.
; RUN: llvm-dis lib.bc -o - | FileCheck %s --check-prefix=PGOName
; PGOName-DAG: define void @_Z7callee1v() {{.*}} !prof ![[#]] {
; PGOName-DAG: define internal void @_ZL7callee0v() {{.*}} !prof ![[#]] !PGOFuncName ![[#MD:]] {
; The source filename of `lib.ll` is specified as "lib.cc" (i.e., the name does
; not change with the directory), so match the full name here.
; PGOName: ![[#MD]] = !{!"lib.cc;_ZL7callee0v"}

; Tests that both external and internal callees are correctly imported.
; RUN: opt -passes=function-import -summary-file summary.thinlto.bc main.bc -o main.import.bc -print-imports 2>&1 | FileCheck %s --check-prefix=IMPORTS
; IMPORTS-DAG: Import _Z7callee1v
; IMPORTS-DAG: Import _ZL7callee0v.llvm.[[#]]
; IMPORTS-DAG: Import _Z11global_funcv

; Tests that ICP transformations happen.
; Both candidates are ICP'ed, check there is no `!VP` in the IR.
; RUN: opt main.import.bc -icp-lto -passes=pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-PROM --implicit-check-not="!VP"
; RUN: opt main.import.bc -icp-lto -passes=pgo-icall-prom -S -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefix=PASS-REMARK

; PASS-REMARK: Promote indirect call to _ZL7callee0v.llvm.[[#]] with count 1 out of 1
; PASS-REMARK: Promote indirect call to _Z7callee1v with count 1 out of 1

; ICALL-PROM:   br i1 %[[#]], label %if.true.direct_targ, label %if.false.orig_indirect, !prof ![[#BRANCH_WEIGHT1:]]
; ICALL-PROM:   br i1 %[[#]], label %if.true.direct_targ1, label %if.false.orig_indirect2, !prof ![[#BRANCH_WEIGHT1]]

; ICALL-PROM: ![[#BRANCH_WEIGHT1]] = !{!"branch_weights", i32 1, i32 0}

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
  call void @_Z11global_funcv()
  ret i32 0
}

declare void @_Z11global_funcv()

;--- lib.ll
source_filename = "lib.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@calleeAddrs = global [2 x ptr] [ptr @_ZL7callee0v, ptr @_Z7callee1v]

define void @_Z7callee1v() {
  ret void
}

define internal void @_ZL7callee0v() {
  ret void
}

define void @_Z11global_funcv() {
entry:
  %0 = load ptr, ptr @calleeAddrs
  call void %0()
  %1 = load ptr, ptr getelementptr inbounds ([2 x ptr], ptr @calleeAddrs, i64 0, i64 1)
  call void %1()
  ret void
}
