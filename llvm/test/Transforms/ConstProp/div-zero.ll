; RUN: opt < %s -instcombine -S | grep "ret i32 0"
; PR4424
declare void @ext()

define i32 @foo(i32 %ptr) {
entry:
        %zero = sub i32 %ptr, %ptr              ; <i32> [#uses=1]
        %div_zero = sdiv i32 %zero, ptrtoint (i32* getelementptr (i32* null,
i32 1) to i32)             ; <i32> [#uses=1]
        ret i32 %div_zero
}

