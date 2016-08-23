; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s --check-prefix=GEN --check-prefix=GEN-COMDAT
; RUN: opt < %s -mtriple=x86_64-apple-darwin -pgo-instr-gen -S | FileCheck %s --check-prefix=GEN --check-prefix=GEN-DARWIN-LINKONCE

; New PM
; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefix=GEN --check-prefix=GEN-COMDAT
; RUN: opt < %s -mtriple=x86_64-apple-darwin -passes=pgo-instr-gen -S | FileCheck %s --check-prefix=GEN --check-prefix=GEN-DARWIN-LINKONCE

; RUN: llvm-profdata merge %S/Inputs/branch1.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE

; RUN: llvm-profdata merge %S/Inputs/branch1_large_count.proftext -o %t.l.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.l.profdata -S | FileCheck %s --check-prefix=USE-LARGE

; New PM
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.l.profdata -S | FileCheck %s --check-prefix=USE-LARGE

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
; GEN-DARWIN-LINKONCE: target triple = "x86_64-apple-darwin"

; GEN-COMDAT: $__llvm_profile_raw_version = comdat any
; GEN-COMDAT: @__llvm_profile_raw_version = constant i64 {{[0-9]+}}, comdat
; GEN-LINKONCE: @__llvm_profile_raw_version = linkonce constant i64 {{[0-9]+}}
; GEN: @__profn_test_br_1 = private constant [9 x i8] c"test_br_1"

define i32 @test_br_1(i32 %i) {
; USE-LABEL: @test_br_1
; USE-SAME: !prof ![[FUNC_ENTRY_COUNT:[0-9]+]]
entry:
; GEN: entry:
; GEN-NOT: llvm.instrprof.increment
  %cmp = icmp sgt i32 %i, 0
  br i1 %cmp, label %if.then, label %if.end
; USE: br i1 %cmp, label %if.then, label %if.end
; USE-SAME: !prof ![[BW_ENTRY:[0-9]+]]
; USE-DAG: ![[BW_ENTRY]] = !{!"branch_weights", i32 2, i32 1}
; USE-LARGE: br i1 %cmp, label %if.then, label %if.end
; USE-LARGE-SAME: !prof ![[BW_L_ENTRY:[0-9]+]]
; USE-LARGE-DAG: ![[BW_L_ENTRY]] = !{!"branch_weights", i32 -1431655766, i32 1431655765}

if.then:
; GEN: if.then:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @__profn_test_br_1, i32 0, i32 0), i64 25571299074, i32 2, i32 1)
  %add = add nsw i32 %i, 2
  br label %if.end

if.end:
; GEN: if.end:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @__profn_test_br_1, i32 0, i32 0), i64 25571299074, i32 2, i32 0)
  %retv = phi i32 [ %add, %if.then ], [ %i, %entry ]
  ret i32 %retv
}
; USE-DAG: {{![0-9]+}} = !{i32 1, !"ProfileSummary", {{![0-9]+}}}
; USE-DAG: {{![0-9]+}} = !{!"DetailedSummary", {{![0-9]+}}}
; USE-DAG: ![[FUNC_ENTRY_COUNT]] = !{!"function_entry_count", i64 3}
