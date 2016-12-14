; RUN: opt < %s -adce -disable-output
; RUN: opt < %s -adce -disable-output -adce-remove-loops

define void @test() {
        br label %BB3

BB3:            ; preds = %BB3, %0
        br label %BB3
}

