; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: llvm-profdata merge %S/Inputs/switch.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; GEN: $__llvm_profile_raw_version = comdat any
; GEN: @__llvm_profile_raw_version = constant i64 72057594037927939, comdat
; GEN: @__profn_test_switch = private constant [11 x i8] c"test_switch"

define void @test_switch(i32 %i) {
entry:
; GEN: entry:
; GEN-NOT: call void @llvm.instrprof.increment
  switch i32 %i, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb2
  ]
; USE: ]
; USE-SAME: !prof ![[BW_SWITCH:[0-9]+]]
; USE: ![[BW_SWITCH]] = !{!"branch_weights", i32 3, i32 2, i32 0, i32 5}

sw.bb:
; GEN: sw.bb:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @__profn_test_switch, i32 0, i32 0), i64 46200943743, i32 4, i32 2)
  br label %sw.epilog

sw.bb1:
; GEN: sw.bb1:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @__profn_test_switch, i32 0, i32 0), i64 46200943743, i32 4, i32 0)
  br label %sw.epilog

sw.bb2:
; GEN: sw.bb2:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @__profn_test_switch, i32 0, i32 0), i64 46200943743, i32 4, i32 1)
  br label %sw.epilog

sw.default:
; GEN: sw.default:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @__profn_test_switch, i32 0, i32 0), i64 46200943743, i32 4, i32 3)
  br label %sw.epilog

sw.epilog:
; GEN: sw.epilog:
; GEN-NOT: call void @llvm.instrprof.increment
  ret void
; GEN: ret void
}
