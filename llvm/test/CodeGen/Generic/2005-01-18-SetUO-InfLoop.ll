; RUN: llvm-as < %s | llc

define void @intersect_pixel() {
entry:
        %tmp125 = fcmp uno double 0.000000e+00, 0.000000e+00            ; <i1> [#uses=1]
        %tmp126 = or i1 %tmp125, false          ; <i1> [#uses=1]
        %tmp126.not = xor i1 %tmp126, true              ; <i1> [#uses=1]
        %brmerge1 = or i1 %tmp126.not, false            ; <i1> [#uses=1]
        br i1 %brmerge1, label %bb154, label %cond_false133

cond_false133:          ; preds = %entry
        ret void

bb154:          ; preds = %entry
        %tmp164 = icmp eq i32 0, 0              ; <i1> [#uses=0]
        ret void
}

declare i1 @llvm.isunordered.f64(double, double)

