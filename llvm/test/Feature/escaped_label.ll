; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

define i32 @foo() {
        br label %"foo`~!@#$%^&*()-_=+{}[]\\\\|;:',<.>/?"

"foo`~!@#$%^&*()-_=+{}[]\\\\|;:',<.>/?":                ; preds = %0
        ret i32 17
}

