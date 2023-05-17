; RUN: opt < %s -passes=pgo-instr-gen -pgo-instrument-entry -S | FileCheck %s --check-prefix=GEN
; RUN: opt < %s -passes=pgo-instr-gen,instrprof -pgo-instrument-entry -atomic-first-counter -S | FileCheck %s --check-prefix=GENA

; RUN: llvm-profdata merge %S/Inputs/branch2.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -pgo-instrument-entry -S | FileCheck %s --check-prefix=USE
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; GEN: $__llvm_profile_raw_version = comdat any
; GEN: @__llvm_profile_raw_version = hidden constant i64 {{[0-9]+}}, comdat
; GEN: @__profn_test_br_2 = private constant [9 x i8] c"test_br_2"

define i32 @test_br_2(i32 %i) {
entry:
; GEN: entry:
; GEN: call void @llvm.instrprof.increment(ptr @__profn_test_br_2, i64 {{[0-9]+}}, i32 2, i32 0)
; GENA: entry:
; GENA: %{{[0-9+]}} = atomicrmw add ptr @__profc_test_br_2, i64 1 monotonic
; USE: br i1 %cmp, label %if.then, label %if.else
; USE-SAME: !prof ![[BW_ENTRY:[0-9]+]]
; USE: ![[BW_ENTRY]] = !{!"branch_weights", i32 0, i32 1}
  %cmp = icmp sgt i32 %i, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
; GEN: if.then:
; GEN-NOT: llvm.instrprof.increment
  %add = add nsw i32 %i, 2
  br label %if.end

if.else:
; GEN: if.else:
; GEN: call void @llvm.instrprof.increment(ptr @__profn_test_br_2, i64 {{[0-9]+}}, i32 2, i32 1)
; GENA: if.else:
; GENA:  %pgocount = load i64, ptr getelementptr inbounds ([2 x i64], ptr @__profc_test_br_2, i32 0, i32 1), align 8
; GENA:  [[V:%[0-9]*]] = add i64 %pgocount, 1
; GENA:  store i64 [[V]], ptr getelementptr inbounds ([2 x i64], ptr @__profc_test_br_2, i32 0, i32 1), align 8
  %sub = sub nsw i32 %i, 2
  br label %if.end

if.end:
; GEN: if.end:
; GEN-NOT: llvm.instrprof.increment
  %retv = phi i32 [ %add, %if.then ], [ %sub, %if.else ]
  ret i32 %retv
; GEN: ret
}
