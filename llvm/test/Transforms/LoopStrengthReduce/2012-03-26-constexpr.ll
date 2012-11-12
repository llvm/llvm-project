; RUN: opt < %s -loop-reduce -S
; PR11950: isHighCostExpansion crashes on ConstExpr
;
; The crash happened during IVChain analysis (CollectChains). We don't
; really care how LSR decides to transform this loop, so we don't
; check it. As long as the analysis doesn't crash we're ok.
target datalayout = "e-p:64:64:64-n32:64"

%struct.this_structure_s.0.5 = type { [6144 x [8 x i32]], [6144 x [8 x i32]], [6147 x [4 x i32]], [8 x i32], [2 x i8*], [2 x i8*], [6144 x i8], [6144 x i32], [6144 x i32], [4 x [4 x i8]] }

define internal fastcc void @someFunction(%struct.this_structure_s.0.5* nocapture %scratch, i32 %stage, i32 %cbSize) nounwind {
entry:
  %0 = getelementptr inbounds %struct.this_structure_s.0.5* %scratch, i32 0, i32 4, i32 %stage
  %1 = load i8** %0, align 4
  %2 = getelementptr inbounds %struct.this_structure_s.0.5* %scratch, i32 0, i32 5, i32 %stage
  %3 = load i8** %2, align 4
  %4 = getelementptr inbounds %struct.this_structure_s.0.5* %scratch, i32 0, i32 2, i32 0, i32 0
  %tmp11 = shl i32 %stage, 1
  %tmp1325 = or i32 %tmp11, 1
  br label %__label_D_1608

__label_D_1608:                                   ; preds = %__label_D_1608, %entry
  %i.12 = phi i32 [ 0, %entry ], [ %10, %__label_D_1608 ]
  %tmp = shl i32 %i.12, 2
  %lvar_g.13 = getelementptr i32* %4, i32 %tmp
  %tmp626 = or i32 %tmp, 1
  %scevgep = getelementptr i32* %4, i32 %tmp626
  %tmp727 = or i32 %tmp, 2
  %scevgep8 = getelementptr i32* %4, i32 %tmp727
  %tmp928 = or i32 %tmp, 3
  %scevgep10 = getelementptr i32* %4, i32 %tmp928
  %scevgep12 = getelementptr %struct.this_structure_s.0.5* %scratch, i32 0, i32 9, i32 %tmp11, i32 %i.12
  %scevgep14 = getelementptr %struct.this_structure_s.0.5* %scratch, i32 0, i32 9, i32 %tmp1325, i32 %i.12
  %5 = load i8* %scevgep12, align 1
  %6 = sext i8 %5 to i32
  %7 = load i8* %scevgep14, align 1
  %8 = sext i8 %7 to i32
  store i32 0, i32* %lvar_g.13, align 4
  store i32 %8, i32* %scevgep, align 4
  store i32 %6, i32* %scevgep8, align 4
  %9 = add nsw i32 %8, %6
  store i32 %9, i32* %scevgep10, align 4
  %10 = add nsw i32 %i.12, 1
  %exitcond = icmp eq i32 %10, 3
  br i1 %exitcond, label %return, label %__label_D_1608

return:                                           ; preds = %__label_D_1608
  ret void
}
