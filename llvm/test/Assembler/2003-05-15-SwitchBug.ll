; RUN: llvm-as %s -disable-output
; RUN: verify-uselistorder %s

; Check minimal switch statement

define void @test(i32 %X) {
        switch i32 %X, label %dest [
        ]

dest:           ; preds = %0
        ret void
}
