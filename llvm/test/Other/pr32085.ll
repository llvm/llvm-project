; RUN: opt -S -O1 < %s -o %t1.ll
; RUN: opt -S < %t1.ll -o %t2.ll
; RUN: opt -S -passes=simplifycfg < %t1.ll -o %t3.ll
;; Show that there's no difference after running another simplify CFG
; RUN: diff %t2.ll %t3.ll

; Test from LoopSink pass, leaves some single-entry single-exit basic blocks.
; After LoopSink, we get a basic block .exit.loopexit which has one entry and
; one exit, the only instruction is a branch. Make sure it doesn't show up.
; Make sure they disappear at -O1.

@g = global i32 0, align 4

define i32 @t1(i32, i32) {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %.exit, label %.preheader

.preheader:
  %invariant = load i32, i32* @g
  br label %.b1

.b1:
  %iv = phi i32 [ %t7, %.b7 ], [ 0, %.preheader ]
  %c1 = icmp sgt i32 %iv, %0
  br i1 %c1, label %.b2, label %.b6

.b2:
  %c2 = icmp sgt i32 %iv, 1
  br i1 %c2, label %.b3, label %.b4

.b3:
  %t3 = sub nsw i32 %invariant, %iv
  br label %.b5

.b4:
  %t4 = add nsw i32 %invariant, %iv
  br label %.b5

.b5:
  %p5 = phi i32 [ %t3, %.b3 ], [ %t4, %.b4 ]
  %t5 = mul nsw i32 %p5, 5
  br label %.b7

.b6:
  %t6 = add nsw i32 %iv, 100
  br label %.b7

.b7:
  %p7 = phi i32 [ %t6, %.b6 ], [ %t5, %.b5 ]
  %t7 = add nuw nsw i32 %iv, 1
  %c7 = icmp eq i32 %t7, %p7
  br i1 %c7, label %.b1, label %.exit

.exit:
  ret i32 10
}

