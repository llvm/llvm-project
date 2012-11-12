; RUN: opt < %s -block-placement -disable-output -print-function 2> /dev/null

define i32 @test() {
        br i1 true, label %X, label %Y

A:              ; preds = %Y, %X
        ret i32 0

X:              ; preds = %0
        br label %A

Y:              ; preds = %0
        br label %A
}

