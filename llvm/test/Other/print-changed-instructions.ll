; RUN: opt -passes=instcombine -disable-output -print-changed=inst-quiet %s 2>&1 | FileCheck %s --check-prefix=CHANGED
; RUN: opt -passes='no-op-function,instcombine' -disable-output -print-changed=inst-quiet %s 2>&1 | FileCheck %s --check-prefix=CHANGED
; RUN: opt -passes=instcombine -disable-output -print-changed=inst-quiet %s > %t.normal 2>&1
; RUN: opt -passes=instcombine -disable-output -print-changed=inst-quiet -print-before-changed %s > %t.before 2>&1
; RUN: diff %t.normal %t.before
; RUN: opt -passes=no-op-function -disable-output -print-changed=inst-quiet %s 2>&1 | FileCheck %s --check-prefix=QUIET --allow-empty
; RUN: opt -passes=no-op-function -disable-output -print-changed=inst %s 2>&1 | FileCheck %s --check-prefix=VERBOSE
; RUN: opt -passes=instcombine -disable-output -print-changed=inst-quiet -filter-passes=NoOpFunctionPass %s 2>&1 | FileCheck %s --check-prefix=QUIET --allow-empty
; RUN: opt -passes=jump-threading -disable-output -print-changed=inst-quiet -filter-print-funcs=move %s 2>&1 | FileCheck %s --check-prefix=MOVE
; RUN: opt -passes=break-crit-edges -disable-output -print-changed=inst-quiet -filter-print-funcs=split %s 2>&1 | FileCheck %s --check-prefix=SPLIT
; RUN: opt -passes=instcombine -disable-output -print-changed=inst-quiet -print-module-scope %s 2>&1 | FileCheck %s --check-prefix=CHANGED
; RUN: opt -passes=no-op-module -disable-output -print-changed=inst -filter-print-funcs=* %s 2>&1 | FileCheck %s --check-prefix=ALL
; RUN: opt -passes=no-op-module -disable-output -print-changed=inst -filter-print-funcs=named -print-module-scope %s 2>&1 | FileCheck %s --check-prefix=ALL

declare i32 @callee(i32)

define i32 @named(i32 %x) {
entry:
  %add = add i32 %x, 0, !annotation !0
  ret i32 %add
}

define i32 @unnamed(i32 %0) {
entry:
  %1 = add i32 %0, 0
  %2 = call i32 @callee(i32 %0)
  ret i32 %2
}

define void @move(i1 %0) {
  br i1 %0, label %5, label %2

2:
  br i1 false, label %named.block, label %3

3:
  %4 = call i32 @callee(i32 0)
  br label %named.block

named.block:
  br label %5

5:
  ret void
}

define void @split(i1 %cond) {
entry:
  br i1 %cond, label %merge, label %other

other:
  br label %merge

merge:
  ret void
}

!0 = !{!"tracked"}

; CHANGED:      *** IR Instruction Changes After InstCombinePass on named ***
; CHANGED-NEXT: - inst#[[ADD:[0-9]+]] @named block#[[NAMED_BLOCK:[0-9]+]]:0   %add = add i32 %x, 0, !annotation !{{[0-9]+}}
; CHANGED-NEXT: - inst#[[RET:[0-9]+]] @named block#[[NAMED_BLOCK]]:1   ret i32 %add
; CHANGED-NEXT: + inst#[[RET]] @named block#[[NAMED_BLOCK]]:0   ret i32 %x
; CHANGED-NEXT: ; summary: instructions +0 -1 changed 1 moved 0; blocks +0 -0 moved 0
; CHANGED:      *** IR Instruction Changes After InstCombinePass on unnamed ***
; CHANGED-NEXT: - inst#[[DEAD:[0-9]+]] @unnamed block#[[UNNAMED_BLOCK:[0-9]+]]:0   %<[[DEAD]]> = add i32 %<[[ARG:[0-9]+]]>, 0
; CHANGED-NEXT: ; summary: instructions +0 -1 changed 0 moved 0; blocks +0 -0 moved 0

; QUIET-NOT: IR Instruction Changes

; VERBOSE:      *** IR Instruction Snapshot At Start ***
; VERBOSE:      + block#[[INITIAL_BLOCK:[0-9]+]] @named:0
; VERBOSE:      + inst#[[INITIAL_INST:[0-9]+]] @named block#[[INITIAL_BLOCK]]:0   %add = add i32 %x, 0, !annotation !{{[0-9]+}}
; VERBOSE:      ; summary: instructions +14 -0 changed 0 moved 0; blocks +10 -0 moved 0
; VERBOSE:      *** IR Instruction Changes After NoOpFunctionPass on named omitted because no change ***
; VERBOSE:      *** IR Instruction Changes After NoOpFunctionPass on unnamed omitted because no change ***

; ALL:      *** IR Instruction Snapshot At Start ***
; ALL-DAG:  + inst#{{[0-9]+}} @named block#{{[0-9]+}}:0
; ALL-DAG:  + inst#{{[0-9]+}} @unnamed block#{{[0-9]+}}:0
; ALL-DAG:  + inst#{{[0-9]+}} @move block#{{[0-9]+}}:0
; ALL-DAG:  + inst#{{[0-9]+}} @split block#{{[0-9]+}}:0

; MOVE:      *** IR Instruction Changes After JumpThreadingPass on move ***
; MOVE-NEXT: - block#[[REMOVED_BLOCK_1:[0-9]+]] @move:1
; MOVE-NEXT: - block#[[REMOVED_BLOCK_2:[0-9]+]] @move:2
; MOVE:      > inst#[[MOVED:[0-9]+]] @move block#[[OLD_BLOCK:[0-9]+]]:0 -> @move block#[[NEW_BLOCK:[0-9]+]]:0   %<[[MOVED]]> = call i32 @callee(i32 0)
; MOVE-NEXT: ; summary: instructions +0 -2 changed 1 moved 1; blocks +0 -2 moved 0

; SPLIT:      *** IR Instruction Changes After BreakCriticalEdgesPass on split ***
; SPLIT-NEXT: + block#[[ADDED_BLOCK:[0-9]+]] @split:1
; SPLIT:      + inst#[[ADDED_INST:[0-9]+]] @split block#[[ADDED_BLOCK]]:0   br label %merge
; SPLIT-NEXT: ; summary: instructions +1 -0 changed 1 moved 0; blocks +1 -0 moved 0
