; RUN: llc < %s -march=ppc32 -mcpu=g3

define void @foo() {
entry:
        br label %bb

bb:             ; preds = %bb, %entry
        br i1 false, label %bb26, label %bb

bb19:           ; preds = %bb26
        ret void

bb26:           ; preds = %bb
        br i1 false, label %bb30, label %bb19

bb30:           ; preds = %bb26
        br label %bb45

bb45:           ; preds = %bb45, %bb30
        %V.0 = phi <8 x i16> [ %tmp42, %bb45 ], [ zeroinitializer, %bb30 ]     ; <<8 x i16>> [#uses=1]
        %tmp42 = mul <8 x i16> zeroinitializer, %V.0            ; <<8 x i16>> [#uses=1]
        br label %bb45
}
