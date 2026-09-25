; RUN: rm -rf %t && split-file %s %t

; RUN: llvm-profdata merge %t/a.proftext -o %t/a.profdata
; RUN: opt -passes=pgo-instr-use -pgo-test-profile-file=%t/a.profdata -module-summary %t/main.ll -o %t/main.bc
; RUN: opt -passes=pgo-instr-use -pgo-test-profile-file=%t/a.profdata -module-summary %t/lib.ll -o %t/lib.bc
; RUN: llvm-lto -thinlto -o %t/summary %t/main.bc %t/lib.bc

; Test that callee with local linkage has `PGOFuncName` metadata while callee with external doesn't have it.
; RUN: llvm-dis %t/lib.bc -o - | FileCheck %s --check-prefix=PGONAME
; PGONAME-DAG: define void @_Z7callee1v() {{.*}} !prof ![[#]]
; PGONAME-DAG: define internal void @_ZL7callee0v() {{.*}} !prof ![[#]] !guid ![[#]] !PGOFuncName ![[#MD:]]
; PGONAME: ![[#MD]] = !{!"lib.cc;_ZL7callee0v"}

; Tests that both external and internal callees are correctly imported.
; RUN: opt -passes=function-import -summary-file %t/summary.thinlto.bc %t/main.bc -o %t/main.import.bc -print-imports 2>&1 | FileCheck %s --check-prefix=IMPORTS
; IMPORTS-DAG: Import _Z7callee1v
; IMPORTS-DAG: Import _ZL7callee0v.llvm.[[#]]
; IMPORTS-DAG: Import _Z11global_funcv

; Tests that ICP transformations happen.
; Both candidates are ICP'ed, check there is no `!VP` in the IR.
; RUN: opt %t/main.import.bc -icp-lto -passes=pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-PROM --implicit-check-not="!VP"
; RUN: opt %t/main.import.bc -icp-lto -passes=pgo-icall-prom -S -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefix=PASS-REMARK

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

;--- a.proftext
:ir
_Z7callee1v
# Func Hash:
742261418966908927
# Num Counters:
1
# Counter Values:
1

main
# Func Hash:
742261418966908927
# Num Counters:
1
# Counter Values:
1

lib.cc;_ZL7callee0v
# Func Hash:
742261418966908927
# Num Counters:
1
# Counter Values:
1

_Z11global_funcv
# Func Hash:
567090795815895039
# Num Counters:
1
# Counter Values:
1
# Num Value Kinds:
1
# ValueKind = IPVK_IndirectCallTarget:
0
# NumValueSites:
2
1
lib.cc;_ZL7callee0v:1
1
_Z7callee1v:1
