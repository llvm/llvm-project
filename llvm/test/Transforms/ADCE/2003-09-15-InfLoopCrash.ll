; RUN: llvm-as < %s | opt -adce -disable-output

define i32 @main() {
        br label %loop

loop:           ; preds = %loop, %0
        br label %loop
}

