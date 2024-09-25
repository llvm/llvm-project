; Check that flattened profile lowering handles cold subgraphs that end in "unreachable"
; RUN: split-file %s %t
; RUN: llvm-ctxprof-util fromJSON --input=%t/profile.json --output=%t/profile.ctxprofdata
; RUN: opt -passes=ctx-prof-flatten %t/example.ll -use-ctx-profile=%t/profile.ctxprofdata -S -o - | FileCheck %s

; CHECK-LABEL: entry:
; CHECK:          br i1 %t, label %yes, label %no, !prof ![[C1:[0-9]+]]
; CHECK-LABEL: no:
; CHECK-NOT:   !prof
; CHECK-LABEL: no1:
; CHECK-NOT:   !prof
; CHECK-LABEL: no2:
; CHECK-NOT:   !prof
; CHECK-LABEL: yes:
; CHECK:          br i1 %t3, label %yes1, label %yes2, !prof ![[C1]]
; CHECK-NOT:   !prof
; CHECK: ![[C1]] = !{!"branch_weights", i32 6, i32 0}

;--- example.ll
define void @f1(i32 %cond) !guid !0 {
entry:
  call void @llvm.instrprof.increment(ptr @f1, i64 42, i32 42, i32 0)
  %t = icmp eq i32 %cond, 1
  br i1 %t, label %yes, label %no

no:
  %t2 = icmp eq i32 %cond, 2
  br i1 %t2, label %no1, label %no2
no1:
  unreachable
no2:
  call void @llvm.instrprof.increment(ptr @f1, i64 42, i32 42, i32 1)
  unreachable
yes:
  %t3 = icmp eq i32 %cond, 3
  br i1 %t3, label %yes1, label %yes2
yes1:
  br label %exit
yes2:
  call void @llvm.instrprof.increment(ptr @f1, i64 42, i32 42, i32 2)
  %t4 = icmp eq i32 %cond, 4
  br i1 %t4, label %yes3, label %yes4
yes3:
  br label %exit
yes4:
  call void @llvm.instrprof.increment(ptr @f1, i64 42, i32 42, i32 3)
  unreachable
exit:
  ret void
}

!0 = !{i64 1234}

;--- profile.json
[{"Guid":1234, "Counters":[6,0,0,0]}]
