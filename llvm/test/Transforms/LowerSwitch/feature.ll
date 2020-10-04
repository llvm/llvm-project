; RUN: opt < %s -lowerswitch -S | FileCheck %s
; RUN: opt < %s -passes=lowerswitch -S | FileCheck %s

; We have switch on input.
; On output we should got binary comparison tree. Check that all is fine.

;CHECK:     entry:
;CHECK-NEXT:  br label %NodeBlock19

;CHECK:     NodeBlock19:                                      ; preds = %entry
;CHECK-NEXT:  %Pivot20 = icmp slt i32 %tmp158, 10
;CHECK-NEXT:  br i1 %Pivot20, label %NodeBlock5, label %NodeBlock17

;CHECK:     NodeBlock17:                                      ; preds = %NodeBlock19
;CHECK-NEXT:  %Pivot18 = icmp slt i32 %tmp158, 13
;CHECK-NEXT:  br i1 %Pivot18, label %NodeBlock9, label %NodeBlock15

;CHECK:     NodeBlock15:                                      ; preds = %NodeBlock17
;CHECK-NEXT:  %Pivot16 = icmp slt i32 %tmp158, 14
;CHECK-NEXT:  br i1 %Pivot16, label %bb330, label %NodeBlock13

;CHECK:     NodeBlock13:                                      ; preds = %NodeBlock15
;CHECK-NEXT:  %Pivot14 = icmp slt i32 %tmp158, 15
;CHECK-NEXT:  br i1 %Pivot14, label %bb332, label %LeafBlock11

;CHECK:     LeafBlock11:                                      ; preds = %NodeBlock13
;CHECK-NEXT:  %SwitchLeaf12 = icmp eq i32 %tmp158, 15
;CHECK-NEXT:  br i1 %SwitchLeaf12, label %bb334, label %NewDefault

;CHECK:     NodeBlock9:                                       ; preds = %NodeBlock17
;CHECK-NEXT:  %Pivot10 = icmp slt i32 %tmp158, 11
;CHECK-NEXT:  br i1 %Pivot10, label %bb324, label %NodeBlock7

;CHECK:     NodeBlock7:                                       ; preds = %NodeBlock9
;CHECK-NEXT:  %Pivot8 = icmp slt i32 %tmp158, 12
;CHECK-NEXT:  br i1 %Pivot8, label %bb326, label %bb328

;CHECK:     NodeBlock5:                                       ; preds = %NodeBlock19
;CHECK-NEXT:  %Pivot6 = icmp slt i32 %tmp158, 7
;CHECK-NEXT:  br i1 %Pivot6, label %NodeBlock, label %NodeBlock3

;CHECK:     NodeBlock3:                                       ; preds = %NodeBlock5
;CHECK-NEXT:  %Pivot4 = icmp slt i32 %tmp158, 8
;CHECK-NEXT:  br i1 %Pivot4, label %bb, label %NodeBlock1

;CHECK:     NodeBlock1:                                       ; preds = %NodeBlock3
;CHECK-NEXT:  %Pivot2 = icmp slt i32 %tmp158, 9
;CHECK-NEXT:  br i1 %Pivot2, label %bb338, label %bb322

;CHECK:     NodeBlock:                                        ; preds = %NodeBlock5
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
