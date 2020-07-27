; RUN: opt < %s -pgo-instr-gen -pgo-instrument-entry=false -S | FileCheck %s --check-prefixes=GEN,NOTENTRY
; RUN: opt < %s -passes=pgo-instr-gen -pgo-instrument-entry=false -S | FileCheck %s --check-prefixes=GEN,NOTENTRY
; RUN: llvm-profdata merge %S/Inputs/branch2.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-instrument-entry=false -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-use -pgo-instrument-entry=false -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE

; RUN: opt < %s -pgo-instr-gen -pgo-instrument-entry=true -S | FileCheck %s --check-prefixes=GEN,ENTRY
; RUN: opt < %s -passes=pgo-instr-gen -pgo-instrument-entry=true  -S | FileCheck %s --check-prefixes=GEN,ENTRY
; RUN: llvm-profdata merge %S/Inputs/branch2_entry.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use  -pgo-instrument-entry=true -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-use  -pgo-instrument-entry=true -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; GEN: $__llvm_profile_raw_version = comdat any
; GEN: @__llvm_profile_raw_version = constant i64 {{[0-9]+}}, comdat
; GEN: @__profn_test_br_2 = private constant [9 x i8] c"test_br_2"

define i32 @test_br_2(i32 %i) {
entry:
; GEN: entry:
; NOTENTRY-NOT: llvm.instrprof.increment
; ENTRY: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @__profn_test_br_2, i32 0, i32 0), i64 29667547796, i32 2, i32 0)
  %cmp = icmp sgt i32 %i, 0
  br i1 %cmp, label %if.then, label %if.else
; USE: br i1 %cmp, label %if.then, label %if.else
; USE-SAME: !prof ![[BW_ENTRY:[0-9]+]]
; USE: ![[BW_ENTRY]] = !{!"branch_weights", i32 1, i32 1}

if.then:
; GEN: if.then:
; NOTENTRY: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @__profn_test_br_2, i32 0, i32 0), i64 29667547796, i32 2, i32 0)
; ENTRY-NOT: llvm.instrprof.increment
  %add = add nsw i32 %i, 2
  br label %if.end

if.else:
; GEN: if.else:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @__profn_test_br_2, i32 0, i32 0), i64 29667547796, i32 2, i32 1)
  %sub = sub nsw i32 %i, 2
  br label %if.end

if.end:
; GEN: if.end:
; GEN-NOT: llvm.instrprof.increment
  %retv = phi i32 [ %add, %if.then ], [ %sub, %if.else ]
  ret i32 %retv
; GEN: ret
}
