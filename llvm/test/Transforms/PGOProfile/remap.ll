; RUN: llvm-profdata merge %S/Inputs/remap.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -pgo-test-profile-remapping-file=%S/Inputs/remap.map -S | FileCheck %s --check-prefix=USE

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_ZN3foo3barERKN1M1XINS_6detail3quxEEE(i32 %i) {
; USE-LABEL: @_ZN3foo3barERKN1M1XINS_6detail3quxEEE
; USE-SAME: !prof ![[FUNC_ENTRY_COUNT:[0-9]+]]
entry:
  %cmp = icmp sgt i32 %i, 0
  br i1 %cmp, label %if.then, label %if.end
; USE: br i1 %cmp, label %if.then, label %if.end
; USE-SAME: !prof ![[BW_ENTRY:[0-9]+]]

if.then:
  %add = add nsw i32 %i, 2
  br label %if.end

if.end:
  %retv = phi i32 [ %add, %if.then ], [ %i, %entry ]
  ret i32 %retv
}

; USE-DAG: {{![0-9]+}} = !{i32 1, !"ProfileSummary", {{![0-9]+}}}
; USE-DAG: {{![0-9]+}} = !{!"DetailedSummary", {{![0-9]+}}}
; USE-DAG: ![[FUNC_ENTRY_COUNT]] = !{!"function_entry_count", i64 3}
; USE-DAG: ![[BW_ENTRY]] = !{!"branch_weights", i32 2, i32 1}
