; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: llvm-profdata merge %S/Inputs/criticaledge.proftext -o %T/criticaledge.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%T/criticaledge.profdata -S | FileCheck %s --check-prefix=USE
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; GEN: @__llvm_profile_name_test_criticalEdge = private constant [17 x i8] c"test_criticalEdge"
; GEN: @"__llvm_profile_name_<stdin>:bar" = private constant [11 x i8] c"<stdin>:bar"

define i32 @test_criticalEdge(i32 %i, i32 %j) {
entry:
; CHECK: entry:
; GEN-NOT: call void @llvm.instrprof.increment
  switch i32 %i, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb2
    i32 4, label %sw.bb2
; CHECK:    i32 3, label %entry.sw.bb2_crit_edge
; CHECK:    i32 4, label %entry.sw.bb2_crit_edge1
    i32 5, label %sw.bb2
  ]
; USE: ]
; USE-SAME: !prof ![[BW_SWITCH:[0-9]+]]

; CHECK: entry.sw.bb2_crit_edge1:
; GEN:   call void @llvm.instrprof.increment(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @__llvm_profile_name_test_criticalEdge, i32 0, i32 0), i64 82323253069, i32 8, i32 1)
; CHECK:   br label %sw.bb2

; CHECK: entry.sw.bb2_crit_edge:
; GEN:   call void @llvm.instrprof.increment(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @__llvm_profile_name_test_criticalEdge, i32 0, i32 0), i64 82323253069, i32 8, i32 0)
; CHECK:   br label %sw.bb2

sw.bb:
; GEN: sw.bb:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @__llvm_profile_name_test_criticalEdge, i32 0, i32 0), i64 82323253069, i32 8, i32 5)
  %call = call i32 @bar(i32 2)
  br label %sw.epilog

sw.bb1:
; GEN: sw.bb1:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @__llvm_profile_name_test_criticalEdge, i32 0, i32 0), i64 82323253069, i32 8, i32 4)
  %call2 = call i32 @bar(i32 1024)
  br label %sw.epilog

sw.bb2:
; GEN: sw.bb2:
; GEN-NOT: call void @llvm.instrprof.increment
  %cmp = icmp eq i32 %j, 2
  br i1 %cmp, label %if.then, label %if.end
; USE: br i1 %cmp, label %if.then, label %if.end
; USE-SAME: !prof ![[BW_SW_BB2:[0-9]+]]

if.then:
; GEN: if.then:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @__llvm_profile_name_test_criticalEdge, i32 0, i32 0), i64 82323253069, i32 8, i32 2)
  %call4 = call i32 @bar(i32 4)
  br label %return

if.end:
; GEN: if.end:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @__llvm_profile_name_test_criticalEdge, i32 0, i32 0), i64 82323253069, i32 8, i32 3)
  %call5 = call i32 @bar(i32 8)
  br label %sw.epilog

sw.default:
; GEN: sw.default:
; GEN-NOT: call void @llvm.instrprof.increment
  %call6 = call i32 @bar(i32 32)
  %cmp7 = icmp sgt i32 %j, 10
  br i1 %cmp7, label %if.then8, label %if.end9
; USE: br i1 %cmp7, label %if.then8, label %if.end9
; USE-SAME: !prof ![[BW_SW_DEFAULT:[0-9]+]]

if.then8:
; GEN: if.then8:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @__llvm_profile_name_test_criticalEdge, i32 0, i32 0), i64 82323253069, i32 8, i32 7)
  %add = add nsw i32 %call6, 10
  br label %if.end9

if.end9:
; GEN: if.end9:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @__llvm_profile_name_test_criticalEdge, i32 0, i32 0), i64 82323253069, i32 8, i32 6)
  %res.0 = phi i32 [ %add, %if.then8 ], [ %call6, %sw.default ]
  br label %sw.epilog

sw.epilog:
; GEN: sw.epilog:
; GEN-NOT: call void @llvm.instrprof.increment
  %res.1 = phi i32 [ %res.0, %if.end9 ], [ %call5, %if.end ], [ %call2, %sw.bb1 ], [ %call, %sw.bb ]
  br label %return

return:
; GEN: return:
; GEN-NOT: call void @llvm.instrprof.increment
  %retval = phi i32 [ %res.1, %sw.epilog ], [ %call4, %if.then ]
  ret i32 %retval
}

define internal i32 @bar(i32 %i) {
entry:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @"__llvm_profile_name_<stdin>:bar", i32 0, i32 0), i64 12884901887, i32 1, i32 0)
  ret i32 %i
}

; USE: ![[BW_SWITCH]] = !{!"branch_weights", i32 2, i32 1, i32 0, i32 2, i32 1, i32 1}
; USE: ![[BW_SW_BB2]] = !{!"branch_weights", i32 2, i32 2}
; USE: ![[BW_SW_DEFAULT]] = !{!"branch_weights", i32 1, i32 1}
