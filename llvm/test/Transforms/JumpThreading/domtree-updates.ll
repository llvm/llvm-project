; RUN: opt < %s -disable-output -passes='jump-threading,print<domtree>' 2>&1 | FileCheck %s

; REQUIRES: asserts

; The idea behind this test case is to verify that the dominator tree is
; updated in a deterministic way. Optimizations, at least EarlyCSE, are
; iterating the vectors that hold child nodes in the DominatorTree. Thus, the
; end result might differ depending on the order in which nodes are inserted
; in the dominator tree. Unfortunately this test case is quite large, but it
; happened to trigger a non-determinism quite often when being executed
; multipe times (it was possible to see varying results when running the test
; less that 10 times in a row).
; The actual problem was tracked down to llvm::MergeBasicBlockIntoOnlyPred, so
; the important property of the test is probably that it triggers a call to
; that function, and that the PredsOfPredBB set that is used to populate
; Updates for the DomTreeUpdater is populated with more than one entry.

; CHECK:      Inorder Dominator Tree: DFSNumbers invalid: 0 slow queries.
; CHECK-NEXT:   [1] %entry {4294967295,4294967295} [0]
; CHECK-NEXT:     [2] %for.cond1 {4294967295,4294967295} [1]
; CHECK-NEXT:       [3] %for.inc19 {4294967295,4294967295} [2]
; CHECK-NEXT:       [3] %if.then {4294967295,4294967295} [2]
; CHECK-NEXT:         [4] %for.cond5.preheader {4294967295,4294967295} [3]
; CHECK-NEXT:           [5] %cleanup {4294967295,4294967295} [4]
; CHECK-NEXT:             [6] %cleanup16 {4294967295,4294967295} [5]
; CHECK-NEXT:               [7] %unreachable {4294967295,4294967295} [6]
; CHECK-NEXT:               [7] %for.end21 {4294967295,4294967295} [6]
; CHECK-NEXT:           [5] %for.body7 {4294967295,4294967295} [4]
; CHECK-NEXT:             [6] %for.inc {4294967295,4294967295} [5]
; CHECK-NEXT:           [5] %return {4294967295,4294967295} [4]
; CHECK-NEXT:       [3] %cleanup16.thread {4294967295,4294967295} [2]
; CHECK-NEXT:     [2] %infinite.loop {4294967295,4294967295} [1]
; CHECK-NEXT: Roots: %entry


@a = dso_local local_unnamed_addr global i16 0, align 1

; Function Attrs: nounwind
define dso_local i16 @g(i16 %a0, i16 %a1, i16 %a2, i16 %a3) local_unnamed_addr {
entry:
  %tobool.not = icmp eq i16 %a0, 0
  br i1 %tobool.not, label %for.cond1, label %infinite.loop

infinite.loop:                                    ; preds = %infinite.loop, %entry
  br label %infinite.loop

for.cond1:                                        ; preds = %for.inc19, %entry
  %retval.0 = phi i16 [ %retval.3, %for.inc19 ], [ undef, %entry ]
  %i.0 = phi i16 [ %i.3, %for.inc19 ], [ undef, %entry ]
  %tobool2.not = icmp eq i16 %a1, 0
  br i1 %tobool2.not, label %if.end15, label %if.then

if.then:                                          ; preds = %for.cond1
  %tobool3.not = icmp eq i16 %a2, 0
  br i1 %tobool3.not, label %if.end15, label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %if.then
  %tobool8.not = icmp eq i16 %a3, 0
  %tobool6.not31 = icmp eq i16 %i.0, 0
  br i1 %tobool6.not31, label %for.end10, label %for.body7

for.body7:                                        ; preds = %for.inc, %for.cond5.preheader
  %i.132 = phi i16 [ %inc, %for.inc ], [ %i.0, %for.cond5.preheader ]
  br i1 %tobool8.not, label %for.inc, label %cleanup

for.inc:                                          ; preds = %for.body7
  %inc = add i16 %i.132, 1
  %tobool6.not = icmp eq i16 %inc, 0
  br i1 %tobool6.not, label %for.end10, label %for.body7

for.end10:                                        ; preds = %for.inc, %for.cond5.preheader
  %i.1.lcssa = phi i16 [ %i.0, %for.cond5.preheader ], [ 0, %for.inc ]
  %.26 = select i1 %tobool8.not, i32 0, i32 4
  br label %cleanup

cleanup:                                          ; preds = %for.end10, %for.body7
  %i.128 = phi i16 [ %i.1.lcssa, %for.end10 ], [ %i.0, %for.body7 ]
  %retval.1 = phi i16 [ %retval.0, %for.end10 ], [ 1, %for.body7 ]
  %cond = phi i1 [ %tobool8.not, %for.end10 ], [ false, %for.body7 ]
  %cleanup.dest.slot.0 = phi i32 [ %.26, %for.end10 ], [ 1, %for.body7 ]
  br i1 %cond, label %if.end15, label %cleanup16

if.end15:                                         ; preds = %cleanup, %if.then, %for.cond1
  %retval.2 = phi i16 [ %retval.1, %cleanup ], [ %retval.0, %if.then ], [ %retval.0, %for.cond1 ]
  %i.2 = phi i16 [ %i.128, %cleanup ], [ %i.0, %if.then ], [ %i.0, %for.cond1 ]
  store i16 0, ptr @a, align 1
  br label %cleanup16

cleanup16:                                        ; preds = %if.end15, %cleanup
  %retval.3 = phi i16 [ %retval.2, %if.end15 ], [ %retval.1, %cleanup ]
  %i.3 = phi i16 [ %i.2, %if.end15 ], [ %i.128, %cleanup ]
  %cleanup.dest.slot.1 = phi i32 [ 0, %if.end15 ], [ %cleanup.dest.slot.0, %cleanup ]
  switch i32 %cleanup.dest.slot.1, label %unreachable [
    i32 0, label %for.inc19
    i32 1, label %return
    i32 4, label %for.end21
  ]

for.inc19:                                        ; preds = %cleanup16
  br label %for.cond1

for.end21:                                        ; preds = %cleanup16
  br label %return

return:                                           ; preds = %for.end21, %cleanup16
  %retval.4 = phi i16 [ 17, %for.end21 ], [ %retval.3, %cleanup16 ]
  ret i16 %retval.4

unreachable:                                      ; preds = %cleanup16
  unreachable
}
