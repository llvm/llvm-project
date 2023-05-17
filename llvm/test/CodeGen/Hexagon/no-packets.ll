; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that there are no packets with two or more instructions, except
; for the endloop packet.

; This is the expected code:
;
;  p0 = cmp.gt(r3,#0)
;  if (!p0) jump:nt .LBB0_3
;  loop0(.LBB0_2,r3)
;  r3 = memw(r1++#4)
;  r4 = memw(r2++#4)
;  r3 = add(r4,r3)
;  memw(r0++#4) = r3
;  { nop; nop }:endloop0
;  jumpr r31

; CHECK-LABEL: fred:
; CHECK:      {
; CHECK-NEXT:   cmp
; CHECK-NEXT: }
; CHECK:      {
; CHECK-NEXT:   jump
; CHECK-NEXT: }
; CHECK:      {
; CHECK-NEXT:   loop0
; CHECK-NEXT: }
; CHECK:      {
; CHECK-NEXT:   memw
; CHECK-NEXT: }
; CHECK:      {
; CHECK-NEXT:   memw
; CHECK-NEXT: }
; CHECK:      {
; CHECK-NEXT:   add
; CHECK-NEXT: }
; CHECK:      {
; CHECK-NEXT:   memw
; CHECK-NEXT: }
; CHECK:      {
; CHECK-NEXT:   nop
; CHECK-NEXT:   nop
; CHECK-NEXT: }{{[ \t]*}}:endloop0
; CHECK:      {
; CHECK-NEXT:   jumpr r31
; CHECK-NEXT: }

target triple = "hexagon"


define void @fred(ptr nocapture %a0, ptr nocapture readonly %a1, ptr nocapture readonly %a2, i32 %a3) local_unnamed_addr #0 {
b4:
  %v5 = icmp sgt i32 %a3, 0
  br i1 %v5, label %b6, label %b21

b6:                                               ; preds = %b4
  br label %b7

b7:                                               ; preds = %b7, %b6
  %v8 = phi i32 [ %v18, %b7 ], [ 0, %b6 ]
  %v9 = phi ptr [ %v17, %b7 ], [ %a0, %b6 ]
  %v10 = phi ptr [ %v14, %b7 ], [ %a2, %b6 ]
  %v11 = phi ptr [ %v12, %b7 ], [ %a1, %b6 ]
  %v12 = getelementptr inbounds i32, ptr %v11, i32 1
  %v13 = load i32, ptr %v11, align 4
  %v14 = getelementptr inbounds i32, ptr %v10, i32 1
  %v15 = load i32, ptr %v10, align 4
  %v16 = add nsw i32 %v15, %v13
  %v17 = getelementptr inbounds i32, ptr %v9, i32 1
  store i32 %v16, ptr %v9, align 4
  %v18 = add nuw nsw i32 %v8, 1
  %v19 = icmp eq i32 %v18, %a3
  br i1 %v19, label %b20, label %b7

b20:                                              ; preds = %b7
  br label %b21

b21:                                              ; preds = %b20, %b4
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="-packets" }

