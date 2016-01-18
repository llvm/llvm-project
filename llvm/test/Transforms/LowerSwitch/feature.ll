; RUN: opt < %s -lowerswitch -S | FileCheck %s

; We have switch on input.
; On output we should got binary comparison tree. Check that all is fine.

;CHECK:     entry:
;CHECK-NEXT:  br label %NodeBlock.19

;CHECK:     NodeBlock.19:                                      ; preds = %entry
;CHECK-NEXT:  %Pivot.20 = icmp slt i32 %tmp158, 10
;CHECK-NEXT:  br i1 %Pivot.20, label %NodeBlock.5, label %NodeBlock.17

;CHECK:     NodeBlock.17:                                      ; preds = %NodeBlock.19
;CHECK-NEXT:  %Pivot.18 = icmp slt i32 %tmp158, 13
;CHECK-NEXT:  br i1 %Pivot.18, label %NodeBlock.9, label %NodeBlock.15

;CHECK:     NodeBlock.15:                                      ; preds = %NodeBlock.17
;CHECK-NEXT:  %Pivot.16 = icmp slt i32 %tmp158, 14
;CHECK-NEXT:  br i1 %Pivot.16, label %bb330, label %NodeBlock.13

;CHECK:     NodeBlock.13:                                      ; preds = %NodeBlock.15
;CHECK-NEXT:  %Pivot.14 = icmp slt i32 %tmp158, 15
;CHECK-NEXT:  br i1 %Pivot.14, label %bb332, label %LeafBlock.11

;CHECK:     LeafBlock.11:                                      ; preds = %NodeBlock.13
;CHECK-NEXT:  %SwitchLeaf12 = icmp eq i32 %tmp158, 15
;CHECK-NEXT:  br i1 %SwitchLeaf12, label %bb334, label %NewDefault

;CHECK:     NodeBlock.9:                                       ; preds = %NodeBlock.17
;CHECK-NEXT:  %Pivot.10 = icmp slt i32 %tmp158, 11
;CHECK-NEXT:  br i1 %Pivot.10, label %bb324, label %NodeBlock.7

;CHECK:     NodeBlock.7:                                       ; preds = %NodeBlock.9
;CHECK-NEXT:  %Pivot.8 = icmp slt i32 %tmp158, 12
;CHECK-NEXT:  br i1 %Pivot.8, label %bb326, label %bb328

;CHECK:     NodeBlock.5:                                       ; preds = %NodeBlock.19
;CHECK-NEXT:  %Pivot.6 = icmp slt i32 %tmp158, 7
;CHECK-NEXT:  br i1 %Pivot.6, label %NodeBlock, label %NodeBlock.3

;CHECK:     NodeBlock.3:                                       ; preds = %NodeBlock.5
;CHECK-NEXT:  %Pivot.4 = icmp slt i32 %tmp158, 8
;CHECK-NEXT:  br i1 %Pivot.4, label %bb, label %NodeBlock.1

;CHECK:     NodeBlock.1:                                       ; preds = %NodeBlock.3
;CHECK-NEXT:  %Pivot.2 = icmp slt i32 %tmp158, 9
;CHECK-NEXT:  br i1 %Pivot.2, label %bb338, label %bb322

;CHECK:     NodeBlock:                                        ; preds = %NodeBlock.5
;CHECK-NEXT:  %Pivot = icmp slt i32 %tmp158, 0
;CHECK-NEXT:  br i1 %Pivot, label %LeafBlock, label %bb338

;CHECK:     LeafBlock:                                        ; preds = %NodeBlock
;CHECK-NEXT:  %tmp158.off = add i32 %tmp158, 6
;CHECK-NEXT:  %SwitchLeaf = icmp ule i32 %tmp158.off, 4
;CHECK-NEXT:  br i1 %SwitchLeaf, label %bb338, label %NewDefault

define i32 @main(i32 %tmp158) {
entry:

        switch i32 %tmp158, label %bb336 [
                 i32 -2, label %bb338
                 i32 -3, label %bb338
                 i32 -4, label %bb338
                 i32 -5, label %bb338
                 i32 -6, label %bb338
                 i32 0, label %bb338
                 i32 1, label %bb338
                 i32 2, label %bb338
                 i32 3, label %bb338
                 i32 4, label %bb338
                 i32 5, label %bb338
                 i32 6, label %bb338
                 i32 7, label %bb
                 i32 8, label %bb338
                 i32 9, label %bb322
                 i32 10, label %bb324
                 i32 11, label %bb326
                 i32 12, label %bb328
                 i32 13, label %bb330
                 i32 14, label %bb332
                 i32 15, label %bb334
        ]
bb:
  ret i32 2
bb322:
  ret i32 3
bb324:
  ret i32 4
bb326:
  ret i32 5
bb328:
  ret i32 6
bb330:
  ret i32 7
bb332:
  ret i32 8
bb334:
  ret i32 9
bb336:
  ret i32 10
bb338:
  ret i32 11
}
