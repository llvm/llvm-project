; RUN: opt < %s -passes=pseudo-probe,sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-profile-metadata.prof -sample-profile-use-profi=0 -S | FileCheck %s

; The test verifies the presence of prof metadata for BranchInst, SwitchInst,
; and IndirectBrInst

@yydebug = dso_local global i32 0, align 4

define dso_local i32 @foo() #0 {
entry:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1)
  %0 = load i32, ptr @yydebug, align 4
  %cmp = icmp ne i32 %0, 0
  br i1 %cmp, label %b1, label %exit
; CHECK: br i1 %cmp, label %b1, label %exit, !prof ![[ENTRY_PROF:[0-9]+]]

b1:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1)
  %1 = load i32, ptr @yydebug, align 4
  switch i32 %1, label %b3 [
    i32 124, label %indirectgoto
    i32 92, label %b2
  ]
; CHECK: ], !prof ![[SWITCH_PROF:[0-9]+]]

b2:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 3, i32 0, i64 -1)
  br label %indirectgoto

b3:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1)
  %2 = load i32, ptr @yydebug, align 4
  ret i32 %2

indirectgoto:
  %indirect.goto.dest = alloca i8, align 4
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 5, i32 0, i64 -1)
  indirectbr ptr %indirect.goto.dest, [label %b1, label %b3, label %b2]
; CHECK: indirectbr ptr %indirect.goto.dest, [label %b1, label %b3, label %b2], !prof ![[GOTO_PROF:[0-9]+]]

exit:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 6, i32 0, i64 -1)
  %3 = load i32, ptr @yydebug, align 4
  ret i32 %3

}

attributes #0 = {"use-sample-profile"}
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #1
!llvm.pseudo_probe_desc = !{!4496}
!4496 = !{i64 6699318081062747564, i64 158517001042, !"foo", null}

; CHECK: ![[ENTRY_PROF]] = !{!"branch_weights", i32 10, i32 6}
; CHECK: ![[SWITCH_PROF]] = !{!"branch_weights", i32 1, i32 9536, i32 1}
; CHECK: ![[GOTO_PROF]] = !{!"branch_weights", i32 17739, i32 1, i32 1}
