; RUN: opt < %s -adce -disable-output

define void @test() {
        br label %BB3

BB3:            ; preds = %BB3, %0
        br label %BB3
}

