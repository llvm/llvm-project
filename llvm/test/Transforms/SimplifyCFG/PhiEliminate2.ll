; RUN: opt < %s -simplifycfg -S | FileCheck %s

define i32 @FoldTwoEntryPHINode(i1 %C, i32 %V1, i32 %V2, i16 %V3) {
entry:
        br i1 %C, label %then, label %else
then:
        %V4 = or i32 %V2, %V1
        br label %Cont
else:
        %V5 = sext i16 %V3 to i32
        br label %Cont
Cont:
        %V6 = phi i32 [ %V5, %else ], [ %V4, %then ]
        call i32 @FoldTwoEntryPHINode( i1 false, i32 0, i32 0, i16 0 )
        ret i32 %V1

; CHECK-LABEL: @FoldTwoEntryPHINode(
; CHECK-NEXT:  entry:
; CHECK-NEXT:  %V5 = sext i16 %V3 to i32
; CHECK-NEXT:  %V4 = or i32 %V2, %V1
; CHECK-NEXT:  %V6 = select i1 %C, i32 %V4, i32 %V5
; CHECK-NEXT:  %0 = call i32 @FoldTwoEntryPHINode(i1 false, i32 0, i32 0, i16 0)
; CHECK-NEXT:  ret i32 %V1
}

