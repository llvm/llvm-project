; RUN: llvm-as < %s | llc
; This crashed the PPC backend.

define void @test() {
        %tmp125 = fcmp uno double 0.000000e+00, 0.000000e+00            ; <i1> [#uses=1]
        br i1 %tmp125, label %bb154, label %cond_false133

cond_false133:          ; preds = %0
        ret void

bb154:          ; preds = %0
        %tmp164 = icmp eq i32 0, 0              ; <i1> [#uses=0]
        ret void
}

