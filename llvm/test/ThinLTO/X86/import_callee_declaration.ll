; RUN: rm -rf %t && split-file %s %t && cd %t

; Generate per-module summaries.
; RUN: opt -module-summary main.ll -o main.bc
; RUN: opt -module-summary lib.ll -o lib.bc

; Generate the combined summary using per-module summaries.
; For function import, set 'import-instr-limit' to 5 and fall back to import
; function declarations.
;
; RUN: llvm-lto2 run \
; RUN:   -import-instr-limit=5 \
; RUN:   -import-declaration \
; RUN:   -thinlto-distributed-indexes \
; RUN:   -thinlto-emit-imports \
; RUN:   -r=main.bc,main,px \
; RUN:   -r=main.bc,small_func, \
; RUN:   -r=main.bc,large_func, \
; RUN:   -r=lib.bc,callee,px \
; RUN:   -r=lib.bc,large_indirect_callee,px \
; RUN:   -r=lib.bc,small_func,px \
; RUN:   -r=lib.bc,large_func,px \
; RUN:   -r=lib.bc,calleeAddrs,px -o summary main.bc lib.bc

; main.ll should import {large_func, large_indirect_callee} as declarations.
; 
; First disassemble per-module summary and find out the GUID for {large_func, large_indirect_callee}.
;
; RUN: llvm-dis lib.bc -o - | FileCheck %s --check-prefix=LIB-DIS
; LIB-DIS: [[LIBMOD:\^[0-9]+]] = module: (path: "lib.bc", hash: (0, 0, 0, 0, 0))
; LIB-DIS: [[LARGEFUNC:\^[0-9]+]] = gv: (name: "large_func", summaries: {{.*}}) ; guid = 2418497564662708935
; LIB-DIS: [[LARGEINDIRECT:\^[0-9]+]] = gv: (name: "large_indirect_callee", summaries: {{.*}}) ; guid = 14343440786664691134
;
; Secondly disassemble main's combined summary and verify the import type of
; these two GUIDs are declaration.
;
; RUN: llvm-dis main.bc.thinlto.bc -o - | FileCheck %s --check-prefix=MAIN-DIS
;
; MAIN-DIS: [[LIBMOD:\^[0-9]+]] = module: (path: "lib.bc", hash: (0, 0, 0, 0, 0))
; MAIN-DIS: [[LARGEFUNC:\^[0-9]+]] = gv: (guid: 2418497564662708935, summaries: (function: (module: [[LIBMOD]], flags: ({{.*}} importType: declaration), insts: 6, {{.*}})))
; MAIN-DIS: [[LARGEINDIRECT:\^[0-9]+]] = gv: (guid: 14343440786664691134, summaries: (function: (module: [[LIBMOD]], flags: ({{.*}} importType: declaration), insts: 7, {{.*}})))

; TODO: Add test coverage for function alias.

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

@calleeAddrs = global [2 x ptr] [ptr @large_indirect_callee, ptr @small_indirect_callee]

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
  ret void
}

define internal void @small_indirect_callee() #0 {
  ret void
}

define void @small_func() {
entry:
  %0 = load ptr, ptr @calleeAddrs
  call void %0(), !prof !0
  %1 = load ptr, ptr getelementptr inbounds ([2 x ptr], ptr @calleeAddrs, i64 0, i64 1)
  call void %1(), !prof !1
  ret void
}

define void @large_func() #0 {
entry:
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
!1 = !{!"VP", i32 0, i64 2, i64 13568239288960714650, i64 1}
