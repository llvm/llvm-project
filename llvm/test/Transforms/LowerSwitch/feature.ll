; RUN: opt < %s -lowerswitch -S | FileCheck %s

; We have switch on input.
; On output we should got binary comparison tree. Check that all is fine.

;CHECK:      entry:
;CHECK-NEXT:   br label %NodeBlock37

;CHECK:      NodeBlock37:                                      ; preds = %entry
;CHECK-NEXT:   %Pivot38 = icmp ult i32 %tmp158, 11
;CHECK-NEXT:   br i1 %Pivot38, label %NodeBlock13, label %NodeBlock35

;CHECK:      NodeBlock35:                                      ; preds = %NodeBlock37
;CHECK-NEXT:   %Pivot36 = icmp ult i32 %tmp158, 14
;CHECK-NEXT:   br i1 %Pivot36, label %NodeBlock23, label %NodeBlock33

;CHECK:      NodeBlock33:                                      ; preds = %NodeBlock35
;CHECK-NEXT:   %Pivot34 = icmp ult i32 %tmp158, 15
;CHECK-NEXT:   br i1 %Pivot34, label %LeafBlock25, label %NodeBlock31

;CHECK:      NodeBlock31:                                      ; preds = %NodeBlock33
;CHECK-NEXT:   %Pivot32 = icmp ult i32 %tmp158, -6
;CHECK-NEXT:   br i1 %Pivot32, label %LeafBlock27, label %LeafBlock29

;CHECK:      LeafBlock29:                                      ; preds = %NodeBlock31
;CHECK-NEXT:   %tmp158.off = add i32 %tmp158, 6
;CHECK-NEXT:   %SwitchLeaf30 = icmp ule i32 %tmp158.off, 4
;CHECK-NEXT:   br i1 %SwitchLeaf30, label %bb338, label %NewDefault

;CHECK:      LeafBlock27:                                      ; preds = %NodeBlock31
;CHECK-NEXT:   %SwitchLeaf28 = icmp eq i32 %tmp158, 15
;CHECK-NEXT:   br i1 %SwitchLeaf28, label %bb334, label %NewDefault

;CHECK:      LeafBlock25:                                      ; preds = %NodeBlock33
;CHECK-NEXT:   %SwitchLeaf26 = icmp eq i32 %tmp158, 14
;CHECK-NEXT:   br i1 %SwitchLeaf26, label %bb332, label %NewDefault

;CHECK:      NodeBlock23:                                      ; preds = %NodeBlock35
;CHECK-NEXT:   %Pivot24 = icmp ult i32 %tmp158, 12
;CHECK-NEXT:   br i1 %Pivot24, label %LeafBlock15, label %NodeBlock21

;CHECK:      NodeBlock21:                                      ; preds = %NodeBlock23
;CHECK-NEXT:   %Pivot22 = icmp ult i32 %tmp158, 13
;CHECK-NEXT:   br i1 %Pivot22, label %LeafBlock17, label %LeafBlock19

;CHECK:      LeafBlock19:                                      ; preds = %NodeBlock21
;CHECK-NEXT:   %SwitchLeaf20 = icmp eq i32 %tmp158, 13
;CHECK-NEXT:   br i1 %SwitchLeaf20, label %bb330, label %NewDefault

;CHECK:      LeafBlock17:                                      ; preds = %NodeBlock21
;CHECK-NEXT:   %SwitchLeaf18 = icmp eq i32 %tmp158, 12
;CHECK-NEXT:   br i1 %SwitchLeaf18, label %bb328, label %NewDefault

;CHECK:      LeafBlock15:                                      ; preds = %NodeBlock23
;CHECK-NEXT:   %SwitchLeaf16 = icmp eq i32 %tmp158, 11
;CHECK-NEXT:   br i1 %SwitchLeaf16, label %bb326, label %NewDefault

;CHECK:      NodeBlock13:                                      ; preds = %NodeBlock37
;CHECK-NEXT:   %Pivot14 = icmp ult i32 %tmp158, 8
;CHECK-NEXT:   br i1 %Pivot14, label %NodeBlock, label %NodeBlock11

;CHECK:      NodeBlock11:                                      ; preds = %NodeBlock13
;CHECK-NEXT:   %Pivot12 = icmp ult i32 %tmp158, 9
;CHECK-NEXT:   br i1 %Pivot12, label %LeafBlock3, label %NodeBlock9

;CHECK:      NodeBlock9:                                       ; preds = %NodeBlock11
;CHECK-NEXT:   %Pivot10 = icmp ult i32 %tmp158, 10
;CHECK-NEXT:   br i1 %Pivot10, label %LeafBlock5, label %LeafBlock7

;CHECK:      LeafBlock7:                                       ; preds = %NodeBlock9
;CHECK-NEXT:   %SwitchLeaf8 = icmp eq i32 %tmp158, 10
;CHECK-NEXT:   br i1 %SwitchLeaf8, label %bb324, label %NewDefault

;CHECK:      LeafBlock5:                                       ; preds = %NodeBlock9
;CHECK-NEXT:   %SwitchLeaf6 = icmp eq i32 %tmp158, 9
;CHECK-NEXT:   br i1 %SwitchLeaf6, label %bb322, label %NewDefault

;CHECK:      LeafBlock3:                                       ; preds = %NodeBlock11
;CHECK-NEXT:   %SwitchLeaf4 = icmp eq i32 %tmp158, 8
;CHECK-NEXT:   br i1 %SwitchLeaf4, label %bb338, label %NewDefault

;CHECK:      NodeBlock:                                        ; preds = %NodeBlock13
;CHECK-NEXT:   %Pivot = icmp ult i32 %tmp158, 7
;CHECK-NEXT:   br i1 %Pivot, label %LeafBlock, label %LeafBlock1

;CHECK:      LeafBlock1:                                       ; preds = %NodeBlock
;CHECK-NEXT:   %SwitchLeaf2 = icmp eq i32 %tmp158, 7
;CHECK-NEXT:   br i1 %SwitchLeaf2, label %bb, label %NewDefault

;CHECK:      LeafBlock:                                        ; preds = %NodeBlock
;CHECK-NEXT:   %SwitchLeaf = icmp ule i32 %tmp158, 6
;CHECK-NEXT:   br i1 %SwitchLeaf, label %bb338, label %NewDefault

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
