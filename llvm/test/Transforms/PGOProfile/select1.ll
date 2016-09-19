; RUN: opt < %s -pgo-instr-gen -pgo-instr-select=true -S | FileCheck %s --check-prefix=GEN
; RUN: opt < %s -passes=pgo-instr-gen -pgo-instr-select=true -S | FileCheck %s --check-prefix=GEN
; RUN: opt < %s -pgo-instr-gen -pgo-instr-select=false -S | FileCheck %s --check-prefix=NOSELECT
; RUN: opt < %s -passes=pgo-instr-gen -pgo-instr-select=false -S | FileCheck %s --check-prefix=NOSELECT
; RUN: llvm-profdata merge %S/Inputs/select1.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -pgo-instr-select=true -S | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -pgo-instr-select=true -S | FileCheck %s --check-prefix=USE
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test_br_2(i32 %i) {
entry:
  %cmp = icmp sgt i32 %i, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %add = add nsw i32 %i, 2
;GEN: %[[STEP:[0-9]+]] = zext i1 %cmp to i64
;GEN: call void @llvm.instrprof.increment.step({{.*}} i32 3, i32 2, i64 %[[STEP]])
;NOSELECT-NOT: call void @llvm.instrprof.increment.step
  %s = select i1 %cmp, i32 %add, i32 0
;USE: select i1 %cmp{{.*}}, !prof ![[BW_ENTRY:[0-9]+]]
;USE: ![[BW_ENTRY]] = !{!"branch_weights", i32 1, i32 3}

  br label %if.end

if.else:
  %sub = sub nsw i32 %i, 2
  br label %if.end

if.end:
  %retv = phi i32 [ %add, %if.then ], [ %sub, %if.else ]
  ret i32 %retv
}
