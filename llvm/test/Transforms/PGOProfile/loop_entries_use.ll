; RUN: rm -rf %t && split-file %s %t

; RUN: llvm-profdata merge %t/default.proftext -o %t/default.profdata
; RUN: opt %t/main.ll -passes=pgo-instr-use -pgo-test-profile-file=%t/default.profdata -S | FileCheck %s
; RUN: llvm-profdata merge %t/loop_entries.proftext -o %t/loop_entries.profdata
; RUN: opt %t/main.ll -passes=pgo-instr-use -pgo-test-profile-file=%t/loop_entries.profdata -S | FileCheck %s
; RUN: llvm-profdata merge %t/function_entry.proftext -o %t/function_entry.profdata
; RUN: opt %t/main.ll -passes=pgo-instr-use -pgo-test-profile-file=%t/function_entry.profdata -S | FileCheck %s

;--- main.ll

define i32 @test_simple_for_with_bypass(i32 %n) {
; CHECK: define i32 @test_simple_for_with_bypass(i32 %n)
; CHECK-SAME: !prof ![[ENTRY_COUNT:[0-9]*]]
entry:
; CHECK: entry:
  %mask = and i32 %n, 65535
  %skip = icmp eq i32 %mask, 0
  br i1 %skip, label %end, label %for.entry
; CHECK: br i1 %skip, label %end, label %for.entry
; CHECK-SAME: !prof ![[BW_FOR_BYPASS:[0-9]+]]

for.entry:
; CHECK: for.entry:
  br label %for.cond

for.cond:
; CHECK: for.cond:
  %i = phi i32 [ 0, %for.entry ], [ %inc1, %for.inc ]
  %sum = phi i32 [ 1, %for.entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %for.body, label %for.end, !prof !1
; CHECK: br i1 %cmp, label %for.body, label %for.end
; CHECK-SAME: !prof ![[BW_FOR_COND:[0-9]+]]

for.body:
; CHECK: for.body:
  %inc = add nsw i32 %sum, 1
  br label %for.inc

for.inc:
; CHECK: for.inc:
  %inc1 = add nsw i32 %i, 1
  br label %for.cond

for.end:
; CHECK: for.end:
  br label %end

end:
; CHECK: end:
  %final_sum = phi i32 [ %sum, %for.end ], [ 0, %entry ]
  ret i32 %final_sum
}

!1 = !{!"branch_weights", i32 100000, i32 80}

; CHECK: ![[ENTRY_COUNT]] = !{!"function_entry_count", i64 12}
; CHECK: ![[BW_FOR_BYPASS]] = !{!"branch_weights", i32 4, i32 8}
; CHECK: ![[BW_FOR_COND]] = !{!"branch_weights", i32 123456, i32 8}

;--- default.proftext

# :ir is the flag to indicate this is IR level profile.
:ir
test_simple_for_with_bypass
# Func Hash:
536873292337293370
# Num Counters:
3
# Counter Values:
123456
12
8

;--- loop_entries.proftext

# :ir is the flag to indicate this is IR level profile.
:ir
# Always instrument the loop entry blocks
:instrument_loop_entries
test_simple_for_with_bypass
# Func Hash:
536873292337293370
# Num Counters:
3
# Counter Values:
123456
12
8

;--- function_entry.proftext

# :ir is the flag to indicate this is IR level profile.
:ir
# Always instrument the function entry block
:entry_first
test_simple_for_with_bypass
# Func Hash:
536873292337293370
# Num Counters:
3
# Counter Values:
12
123456
8
