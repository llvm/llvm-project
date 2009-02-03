; RUN: llvm-as < %s | llc -march=x86
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9"
        %struct.wxPoint2DInt = type { i32, i32 }

define x86_fp80 @_ZNK12wxPoint2DInt14GetVectorAngleEv(%struct.wxPoint2DInt* %this) {
entry:
        br i1 false, label %cond_true, label %UnifiedReturnBlock

cond_true:              ; preds = %entry
        %tmp8 = load i32* null, align 4         ; <i32> [#uses=1]
        %tmp9 = icmp sgt i32 %tmp8, -1          ; <i1> [#uses=1]
        %retval = select i1 %tmp9, x86_fp80 0xK4005B400000000000000, x86_fp80 0xK40078700000000000000           ; <x86_fp80> [#uses=1]
        ret x86_fp80 %retval

UnifiedReturnBlock:             ; preds = %entry
        ret x86_fp80 0xK4005B400000000000000
}
