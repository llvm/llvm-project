; "-debug-only" requires asserts.
; REQUIRES: asserts
; RUN: rm -rf %t && split-file %s %t && cd %t

; Generate per-module summaries.
; RUN: opt -module-summary main.ll -o main.bc
; RUN: opt -module-summary lib.ll -o lib.bc

; Generate the combined summary and distributed indices.

; - For function import, set 'import-instr-limit' to 7 and fall back to import
;   function declarations.
; - In main.ll, function 'main' calls 'small_func' and 'large_func'. Both callees
;   are defined in lib.ll. 'small_func' has two indirect callees, one is smaller
;   and the other one is larger. Both callees of 'small_func' are defined in lib.ll.
; - Given the import limit, in main's combined summary, the import type of 'small_func'
;   and 'small_indirect_callee' will be 'definition', and the import type of
;   'large_func' and 'large_indirect_callee' will be 'declaration'.
;
; The test will disassemble combined summaries and check the import type is
; correct. Right now postlink optimizer pipeline doesn't do anything (e.g.,
; import the declaration or de-serialize summary attributes yet) so there is
; nothing to test more than the summary content.
;
; RUN: llvm-lto2 run \
; RUN:   -debug-only=function-import \
; RUN:   -import-instr-limit=7 \
; RUN:   -import-declaration \
; RUN:   -thinlto-distributed-indexes \
; RUN:   -r=main.bc,main,px \
; RUN:   -r=main.bc,small_func, \
; RUN:   -r=main.bc,large_func, \
; RUN:   -r=lib.bc,callee,pl \
; RUN:   -r=lib.bc,large_indirect_callee,px \
; RUN:   -r=lib.bc,small_func,px \
; RUN:   -r=lib.bc,large_func,px \
; RUN:   -r=lib.bc,large_indirect_callee_alias,px \
; RUN:   -r=lib.bc,calleeAddrs,px -o summary main.bc lib.bc 2>&1 | FileCheck %s --check-prefix=DUMP
;
; RUN: llvm-lto -thinlto-action=thinlink -import-declaration -import-instr-limit=7  -o combined.index.bc main.bc lib.bc
; RUN: llvm-lto -thinlto-action=distributedindexes -debug-only=function-import -import-declaration -import-instr-limit=7 -thinlto-index combined.index.bc main.bc lib.bc 2>&1 | FileCheck %s --check-prefix=DUMP

; DUMP: - 2 function definitions and 3 function declarations imported from lib.bc

; First disassemble per-module summary and find out the GUID for {large_func, large_indirect_callee}.
;
; RUN: llvm-dis lib.bc -o - | FileCheck %s --check-prefix=LIB-DIS
; LIB-DIS: [[LARGEFUNC:\^[0-9]+]] = gv: (name: "large_func", summaries: {{.*}}) ; guid = 2418497564662708935
; LIB-DIS: [[LARGEINDIRECT:\^[0-9]+]] = gv: (name: "large_indirect_callee", summaries: {{.*}}) ; guid = 14343440786664691134
; LIB-DIS: [[LARGEINDIRECTALIAS:\^[0-9]+]] = gv: (name: "large_indirect_callee_alias", summaries: {{.*}}, aliasee: [[LARGEINDIRECT]]
;
; Secondly disassemble main's combined summary and test that large callees are
; not imported as declarations yet.
;
; RUN: llvm-dis main.bc.thinlto.bc -o - | FileCheck %s --check-prefix=MAIN-DIS
;
; MAIN-DIS: [[LIBMOD:\^[0-9]+]] = module: (path: "lib.bc", hash: (0, 0, 0, 0, 0))
; MAIN-DIS-NOT: [[LARGEFUNC:\^[0-9]+]] = gv: (guid: 2418497564662708935, summaries: (function: (module: [[LIBMOD]], flags: ({{.*}} importType: declaration), insts: 8, {{.*}})))
; MAIN-DIS-NOT: [[LARGEINDIRECT:\^[0-9]+]] = gv: (guid: 14343440786664691134, summaries: (function: (module: [[LIBMOD]], flags: ({{.*}} importType: declaration), insts: 8, {{.*}})))
; MAIN-DIS-NOT: [[LARGEINDIRECTALIAS:\^[0-9]+]] = gv: (guid: 16730173943625350469, summaries: (alias: (module: [[LIBMOD]], flags: ({{.*}} importType: declaration)

; Run in-process ThinLTO and tests that
; 1. `callee` remains internalized even if the symbols of its callers
; (large_func and large_indirect_callee) are exported as declarations and visible to main module.
; 2. the debugging logs from `function-import` pass are expected.

; RUN: llvm-lto2 run \
; RUN:   -debug-only=function-import \
; RUN:   -save-temps \
; RUN:   -thinlto-threads=1 \
; RUN:   -import-instr-limit=7 \
; RUN:   -import-declaration \
; RUN:   -r=main.bc,main,px \
; RUN:   -r=main.bc,small_func, \
; RUN:   -r=main.bc,large_func, \
; RUN:   -r=lib.bc,callee,pl \
; RUN:   -r=lib.bc,large_indirect_callee,px \
; RUN:   -r=lib.bc,small_func,px \
; RUN:   -r=lib.bc,large_func,px \
; RUN:   -r=lib.bc,large_indirect_callee_alias,px \
; RUN:   -r=lib.bc,calleeAddrs,px -o in-process main.bc lib.bc 2>&1 | FileCheck %s --check-prefix=IMPORTDUMP

; Test import status from debugging logs.
; TODO: Serialize declaration bit and test declaration bits are correctly set,
; and extend this test case to test IR once postlink optimizer makes use of
; the import type for declarations.
; IMPORTDUMP-DAG: Not importing function 11825436545918268459 callee from lib.cc
; IMPORTDUMP-DAG: Is importing function declaration 14343440786664691134 large_indirect_callee from lib.cc
; IMPORTDUMP-DAG: Is importing function definition 13568239288960714650 small_indirect_callee from lib.cc
; IMPORTDUMP-DAG: Is importing function definition 6976996067367342685 small_func from lib.cc
; IMPORTDUMP-DAG: Is importing function declaration 2418497564662708935 large_func from lib.cc
; IMPORTDUMP-DAG: Not importing global 7680325410415171624 calleeAddrs from lib.cc
; IMPORTDUMP-DAG: Is importing alias declaration 16730173943625350469 large_indirect_callee_alias from lib.cc

; RUN: llvm-dis in-process.1.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT

; RUN: llvm-dis in-process.2.2.internalize.bc -o - | FileCheck %s --check-prefix=INTERNALIZE

; IMPORT-DAG: define available_externally void @small_func
; IMPORT-DAG: define available_externally hidden void @small_indirect_callee
; IMPORT-DAG: declare void @large_func
; IMPORT-NOT: large_indirect_callee
; IMPORT-NOT: large_indirect_callee_alias

; INTERNALIZE: define internal void @callee()

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
  call void @small_func()
  call void @large_func()
  ret i32 0
}

declare void @small_func()

; large_func without attributes
declare void @large_func()

;--- lib.ll
source_filename = "lib.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@calleeAddrs = global [3 x ptr] [ptr @large_indirect_callee, ptr @small_indirect_callee, ptr @large_indirect_callee_alias]

define void @callee() #1 {
  ret void
}

define void @large_indirect_callee()#2 {
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  ret void
}

define internal void @small_indirect_callee() #0 {
  ret void
}

@large_indirect_callee_alias = alias void(), ptr @large_indirect_callee

define void @small_func() {
entry:
  %0 = load ptr, ptr @calleeAddrs
  call void %0(), !prof !0
  %1 = load ptr, ptr getelementptr inbounds ([3 x ptr], ptr @calleeAddrs, i64 0, i64 1)
  call void %1(), !prof !1
  %2 = load ptr, ptr getelementptr inbounds ([3 x ptr], ptr @calleeAddrs, i64 0, i64 2)
  call void %2(), !prof !2
  ret void
}

define void @large_func() #0 {
entry:
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  ret void
}

attributes #0 = { nounwind norecurse }

attributes #1 = { noinline }

attributes #2 = { norecurse }

!0 = !{!"VP", i32 0, i64 1, i64 14343440786664691134, i64 1}
!1 = !{!"VP", i32 0, i64 1, i64 13568239288960714650, i64 1}
!2 = !{!"VP", i32 0, i64 1, i64 16730173943625350469, i64 1}
