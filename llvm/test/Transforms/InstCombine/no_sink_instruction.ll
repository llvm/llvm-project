; RUN: opt -instcombine -instcombine-code-sinking=0 -S < %s | FileCheck %s

define i32 @test(i1 %C, i32 %A, i32 %B) {
; CHECK-LABEL: @test(
; CHECK: sdiv i32
; CHECK-NEXT: add i32
entry:
        %tmp.2 = sdiv i32 %A, %B                ; <i32> [#uses=1]
        %tmp.9 = add i32 %B, %A         ; <i32> [#uses=1]
        br i1 %C, label %then, label %endif

then:           ; preds = %entry
; CHECK: ret i32
        ret i32 %tmp.9

endif:          ; preds = %entry
; CHECK: ret i32
        ret i32 %tmp.2
}
