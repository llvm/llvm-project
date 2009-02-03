; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx
; PR2574

define void @entry(i32 %m_task_id, i32 %start_x, i32 %end_x) {; <label>:0
        br i1 true, label %bb.nph, label %._crit_edge

bb.nph:         ; preds = %bb.nph, %0
        %t2206f2.0 = phi <2 x float> [ %2, %bb.nph ], [ undef, %0 ]             ; <<2 x float>> [#uses=1]
        insertelement <2 x float> %t2206f2.0, float 0.000000e+00, i32 0         ; <<2 x float>>:1 [#uses=1]
        insertelement <2 x float> %1, float 0.000000e+00, i32 1         ; <<2 x float>>:2 [#uses=1]
        br label %bb.nph

._crit_edge:            ; preds = %0
        ret void
}
