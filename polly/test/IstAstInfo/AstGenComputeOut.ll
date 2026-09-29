; This test checks that Polly's ISL AST generation aborts gracefully when the
; ISL operation quota (set via -polly-astgen-computeout) is exhausted, instead
; of running for an unbounded amount of time.
;
; The SCoP is constructed to be deliberately expensive for AST generation:
;   - a large iteration space (the outer loop runs with trip count 65536, see
;     the exit test 'icmp eq i64 %phi, 65536' in bb49), combined with
;   - a long chain of conditionals (bb9, bb15, bb18, ... bb48), where one
;     branch of every conditional flows into the common block bb7.
; Because bb7 is reached from all of these predecessors, its domain becomes the
; union of every reaching condition, which pushes ISL AST generation into a
; high-dimensional, blow-up search space. Without a bound this does not finish
; in reasonable time.
; This test case has the same characteristics as the one in
; PR https://github.com/llvm/llvm-project/pull/203073, but is effectively a
; "larger problem". The complexity of its structure made this test case bail
; out of the DeLICM phase, yet it previously got stuck in the ISL AST
; generation phase.
;
; The operation limit of polly-astgen-computeout is set to 1 -- the smallest
; value that still arms the guard (0 means unbounded) -- so the quota trips
; as early as possible and the test does not depend on an arbitrary tuned cutoff value.

; REQUIRES: asserts

; RUN: opt %loadNPMPolly %s -passes='polly-custom<ast>' -polly-process-unprofitable \
; RUN:   -polly-astgen-computeout=1 -debug-only=polly-ast \
; RUN:   -disable-output 2>&1 | FileCheck %s

define void @eggs(i32 %arg) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb49, %bb
  %phi = phi i64 [ 1, %bb ], [ %add, %bb49 ]
  br i1 true, label %bb2, label %bb5

bb2:                                              ; preds = %bb1
  %icmp = icmp eq i32 %arg, 0
  br i1 %icmp, label %bb3, label %bb5

bb3:                                              ; preds = %bb2
  %trunc = trunc i64 %phi to i32
  %and = and i32 %trunc, 1
  %icmp4 = icmp eq i32 %and, 0
  br i1 %icmp4, label %bb9, label %bb5

bb5:                                              ; preds = %bb3, %bb2, %bb1
  %phi6 = phi i8 [ 0, %bb2 ], [ 1, %bb1 ], [ 0, %bb3 ]
  br label %bb7

bb7:                                              ; preds = %bb48, %bb45, %bb42, %bb39, %bb36, %bb33, %bb30, %bb27, %bb24, %bb21, %bb18, %bb15, %bb9, %bb5
  %phi8 = phi i8 [ 1, %bb5 ], [ 0, %bb9 ], [ 0, %bb48 ], [ 0, %bb15 ], [ 0, %bb18 ], [ 0, %bb21 ], [ 0, %bb24 ], [ 0, %bb27 ], [ 0, %bb30 ], [ 0, %bb33 ], [ 0, %bb36 ], [ 0, %bb39 ], [ 0, %bb42 ], [ 0, %bb45 ]
  store i8 0, ptr null, align 1
  br label %bb49

bb9:                                              ; preds = %bb3
  %and10 = and i32 %trunc, 2
  %icmp11 = icmp eq i32 %and10, 0
  %and12 = and i32 %trunc, 4
  %icmp13 = icmp eq i32 %and12, 0
  %and14 = and i1 %icmp11, %icmp13
  br i1 %and14, label %bb15, label %bb7

bb15:                                             ; preds = %bb9
  %and16 = and i32 %trunc, 8
  %icmp17 = icmp eq i32 %and16, 0
  br i1 %icmp17, label %bb18, label %bb7

bb18:                                             ; preds = %bb15
  %and19 = and i32 %trunc, 16
  %icmp20 = icmp eq i32 %and19, 0
  br i1 %icmp20, label %bb21, label %bb7

bb21:                                             ; preds = %bb18
  %and22 = and i32 %trunc, 32
  %icmp23 = icmp eq i32 %and22, 0
  br i1 %icmp23, label %bb24, label %bb7

bb24:                                             ; preds = %bb21
  %and25 = and i32 %trunc, 64
  %icmp26 = icmp eq i32 %and25, 0
  br i1 %icmp26, label %bb27, label %bb7

bb27:                                             ; preds = %bb24
  %and28 = and i32 %trunc, 128
  %icmp29 = icmp eq i32 %and28, 0
  br i1 %icmp29, label %bb30, label %bb7

bb30:                                             ; preds = %bb27
  %and31 = and i32 %trunc, 256
  %icmp32 = icmp eq i32 %and31, 0
  br i1 %icmp32, label %bb33, label %bb7

bb33:                                             ; preds = %bb30
  %and34 = and i32 %trunc, 512
  %icmp35 = icmp eq i32 %and34, 0
  br i1 %icmp35, label %bb36, label %bb7

bb36:                                             ; preds = %bb33
  %and37 = and i32 %trunc, 1024
  %icmp38 = icmp eq i32 %and37, 0
  br i1 %icmp38, label %bb39, label %bb7

bb39:                                             ; preds = %bb36
  %and40 = and i32 %trunc, 2048
  %icmp41 = icmp eq i32 %and40, 0
  br i1 %icmp41, label %bb42, label %bb7

bb42:                                             ; preds = %bb39
  %and43 = and i32 %trunc, 4096
  %icmp44 = icmp eq i32 %and43, 0
  br i1 %icmp44, label %bb45, label %bb7

bb45:                                             ; preds = %bb42
  %and46 = and i32 %trunc, 8192
  %icmp47 = icmp eq i32 %and46, 0
  br i1 %icmp47, label %bb48, label %bb7

bb48:                                             ; preds = %bb45
  br i1 false, label %bb49, label %bb7

bb49:                                             ; preds = %bb48, %bb7
  %add = add i64 %phi, 1
  %icmp50 = icmp eq i64 %phi, 65536
  br i1 %icmp50, label %bb51, label %bb1

bb51:                                             ; preds = %bb49
  ret void
}

; CHECK: AST generation for SCoP in function 'eggs' exceeded operation limit (operations). Skipping.
