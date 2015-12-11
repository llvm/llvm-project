; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: llvm-profdata merge %S/Inputs/branch1.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; GEN: @__llvm_profile_name_test_br_1 = private constant [9 x i8] c"test_br_1"

define i32 @test_br_1(i32 %i) {
entry:
; GEN: entry:
; GEN-NOT: llvm.instrprof.increment
  %cmp = icmp sgt i32 %i, 0
  br i1 %cmp, label %if.then, label %if.end
; USE: br i1 %cmp, label %if.then, label %if.end
; USE-SAME: !prof ![[BW_ENTRY:[0-9]+]]
; USE: ![[BW_ENTRY]] = !{!"branch_weights", i32 2, i32 1}

if.then:
; GEN: if.then:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @__llvm_profile_name_test_br_1, i32 0, i32 0), i64 25571299074, i32 2, i32 1)
  %add = add nsw i32 %i, 2
  br label %if.end

if.end:
; GEN: if.end:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @__llvm_profile_name_test_br_1, i32 0, i32 0), i64 25571299074, i32 2, i32 0)
  %retv = phi i32 [ %add, %if.then ], [ %i, %entry ]
  ret i32 %retv
}
