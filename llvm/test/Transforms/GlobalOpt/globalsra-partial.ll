; In this case, the global cannot be merged as i may be out of range

; RUN: opt < %s -passes=globalopt -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@G = internal global { i32, [4 x float] } zeroinitializer               ; <ptr> [#uses=3]

; CHECK: @G = internal unnamed_addr global { i32, [4 x float] }
; CHECK: 12345
define void @onlystore() {
        store i32 12345, ptr getelementptr ({ i32, [4 x float] }, ptr @G, i32 0, i32 0)
        ret void
}

define void @storeinit(i32 %i) {
        %Ptr = getelementptr { i32, [4 x float] }, ptr @G, i32 0, i32 1, i32 %i             ; <ptr> [#uses=1]
        store float 1.000000e+00, ptr %Ptr
        ret void
}

define float @readval(i32 %i) {
        %Ptr = getelementptr { i32, [4 x float] }, ptr @G, i32 0, i32 1, i32 %i             ; <ptr> [#uses=1]
        %V = load float, ptr %Ptr           ; <float> [#uses=1]
        ret float %V
}
