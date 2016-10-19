; RUN: llc < %s -mtriple=armv7-apple-darwin | FileCheck %s
; PHI elimination shouldn't break backedge.
; rdar://8263994

%struct.list_data_s = type { i16, i16 }
%struct.list_head = type { %struct.list_head*, %struct.list_data_s* }

define arm_apcscc %struct.list_head* @t1(%struct.list_head* %list) nounwind {
entry:
; CHECK-LABEL: t1:
  %0 = icmp eq %struct.list_head* %list, null
  br i1 %0, label %bb2, label %bb

bb:
; CHECK: LBB0_2:
; CHECK: bne LBB0_2
; CHECK-NOT: b LBB0_2
; CHECK: bx lr
  %list_addr.05 = phi %struct.list_head* [ %2, %bb ], [ %list, %entry ]
  %next.04 = phi %struct.list_head* [ %list_addr.05, %bb ], [ null, %entry ]
  %1 = getelementptr inbounds %struct.list_head, %struct.list_head* %list_addr.05, i32 0, i32 0
  %2 = load %struct.list_head*, %struct.list_head** %1, align 4
  store %struct.list_head* %next.04, %struct.list_head** %1, align 4
  %3 = icmp eq %struct.list_head* %2, null
  br i1 %3, label %bb2, label %bb

bb2:
  %next.0.lcssa = phi %struct.list_head* [ null, %entry ], [ %list_addr.05, %bb ]
  ret %struct.list_head* %next.0.lcssa
}

; Optimize loop entry, eliminate intra loop branches
; rdar://8117827
define i32 @t2(i32 %passes, i32* nocapture %src, i32 %size) nounwind readonly {
entry:
; CHECK-LABEL: t2:
; CHECK: beq LBB1_[[RET:.]]
  %0 = icmp eq i32 %passes, 0                     ; <i1> [#uses=1]
  br i1 %0, label %bb5, label %bb.nph15

; CHECK: LBB1_[[PREHDR:.]]: @ %bb2.preheader
bb1:                                              ; preds = %bb2.preheader, %bb1
; CHECK: LBB1_[[BB1:.]]: @ %bb1
; CHECK: bne LBB1_[[BB1]]
  %indvar = phi i32 [ %indvar.next, %bb1 ], [ 0, %bb2.preheader ] ; <i32> [#uses=2]
  %sum.08 = phi i32 [ %2, %bb1 ], [ %sum.110, %bb2.preheader ] ; <i32> [#uses=1]
  %tmp17 = sub i32 %i.07, %indvar                 ; <i32> [#uses=1]
  %scevgep = getelementptr i32, i32* %src, i32 %tmp17  ; <i32*> [#uses=1]
  %1 = load i32, i32* %scevgep, align 4                ; <i32> [#uses=1]
  %2 = add nsw i32 %1, %sum.08                    ; <i32> [#uses=2]
  %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %indvar.next, %size     ; <i1> [#uses=1]
  br i1 %exitcond, label %bb3, label %bb1

bb3:                                              ; preds = %bb1, %bb2.preheader
; CHECK: LBB1_[[BB3:.]]: @ %bb3
; CHECK: bne LBB1_[[PREHDR]]
; CHECK-NOT: b LBB1_
  %sum.0.lcssa = phi i32 [ %sum.110, %bb2.preheader ], [ %2, %bb1 ] ; <i32> [#uses=2]
  %3 = add i32 %pass.011, 1                       ; <i32> [#uses=2]
  %exitcond18 = icmp eq i32 %3, %passes           ; <i1> [#uses=1]
  br i1 %exitcond18, label %bb5, label %bb2.preheader

bb.nph15:                                         ; preds = %entry
  %i.07 = add i32 %size, -1                       ; <i32> [#uses=2]
  %4 = icmp sgt i32 %i.07, -1                     ; <i1> [#uses=1]
  br label %bb2.preheader

bb2.preheader:                                    ; preds = %bb3, %bb.nph15
  %pass.011 = phi i32 [ 0, %bb.nph15 ], [ %3, %bb3 ] ; <i32> [#uses=1]
  %sum.110 = phi i32 [ 0, %bb.nph15 ], [ %sum.0.lcssa, %bb3 ] ; <i32> [#uses=2]
  br i1 %4, label %bb1, label %bb3

; CHECK: LBB1_[[RET]]: @ %bb5
; CHECK: pop
bb5:                                              ; preds = %bb3, %entry
  %sum.1.lcssa = phi i32 [ 0, %entry ], [ %sum.0.lcssa, %bb3 ] ; <i32> [#uses=1]
  ret i32 %sum.1.lcssa
}
