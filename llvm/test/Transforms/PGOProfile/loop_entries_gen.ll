; RUN: opt %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefixes=CHECK,NOTLOOPENTRIES --implicit-check-not=@llvm.instrprof.increment
; RUN: opt %s -passes=pgo-instr-gen -pgo-instrument-loop-entries -S | FileCheck %s --check-prefixes=CHECK,LOOPENTRIES --implicit-check-not=@llvm.instrprof.increment
; RUN: opt %s -passes=pgo-instr-gen -pgo-instrument-entry -S | FileCheck %s --check-prefixes=CHECK,FUNCTIONENTRY --implicit-check-not=@llvm.instrprof.increment

; CHECK: $__llvm_profile_raw_version = comdat any
; CHECK: @__llvm_profile_raw_version = hidden constant i64 {{[0-9]+}}, comdat
; CHECK: @__profn_test_simple_for_with_bypass = private constant [27 x i8] c"test_simple_for_with_bypass"

define i32 @test_simple_for_with_bypass(i32 %n) {
entry:
; CHECK: entry:
; NOTLOOPENTRIES: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 1)
; LOOPENTRIES: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 1)
; FUNCTIONENTRY: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 0)
  %mask = and i32 %n, 65535
  %skip = icmp eq i32 %mask, 0
  br i1 %skip, label %end, label %for.entry

for.entry:
; CHECK: for.entry:
; LOOPENTRIES: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 2)
  br label %for.cond

for.cond:
; CHECK: for.cond:
  %i = phi i32 [ 0, %for.entry ], [ %inc1, %for.inc ]
  %sum = phi i32 [ 1, %for.entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %for.body, label %for.end, !prof !1

for.body:
; CHECK: for.body:
  %inc = add nsw i32 %sum, 1
  br label %for.inc

for.inc:
; CHECK: for.inc:
; NOTLOOPENTRIES: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 0)
; LOOPENTRIES: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 0)
; FUNCTIONENTRY: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 1)
  %inc1 = add nsw i32 %i, 1
  br label %for.cond

for.end:
; CHECK: for.end:
; NOTLOOPENTRIES: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 2)
; FUNCTIONENTRY: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 2)
  br label %end

end:
; CHECK: end:
  %final_sum = phi i32 [ %sum, %for.end ], [ 0, %entry ]
  ret i32 %final_sum
}

; CHECK: declare void @llvm.instrprof.increment(ptr, i64, i32, i32) #0

!1 = !{!"branch_weights", i32 100000, i32 80}
