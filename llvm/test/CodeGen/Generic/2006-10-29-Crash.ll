; RUN: llc < %s

define void @form_component_prediction(i32 %dy) {
entry:
        %tmp7 = and i32 %dy, 1          ; <i32> [#uses=1]
        %tmp27 = icmp eq i32 %tmp7, 0           ; <i1> [#uses=1]
        br i1 false, label %cond_next30, label %bb115

cond_next30:            ; preds = %entry
        ret void

bb115:          ; preds = %entry
        %bothcond1 = or i1 %tmp27, false                ; <i1> [#uses=1]
        br i1 %bothcond1, label %bb228, label %cond_next125

cond_next125:           ; preds = %bb115
        ret void

bb228:          ; preds = %bb115
        ret void
}

