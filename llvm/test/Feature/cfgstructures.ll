; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

;; This is an irreducible flow graph
define void @irreducible(i1 %cond) {
        br i1 %cond, label %X, label %Y

X:              ; preds = %Y, %0
        br label %Y

Y:              ; preds = %X, %0
        br label %X
}

;; This is a pair of loops that share the same header
define void @sharedheader(i1 %cond) {
        br label %A

A:              ; preds = %Y, %X, %0
        br i1 %cond, label %X, label %Y

X:              ; preds = %A
        br label %A

Y:              ; preds = %A
        br label %A
}


;; This is a simple nested loop
define void @nested(i1 %cond1, i1 %cond2, i1 %cond3) {
        br label %Loop1

Loop1:          ; preds = %L2Exit, %0
        br label %Loop2

Loop2:          ; preds = %L3Exit, %Loop1
        br label %Loop3

Loop3:          ; preds = %Loop3, %Loop2
        br i1 %cond3, label %Loop3, label %L3Exit

L3Exit:         ; preds = %Loop3
        br i1 %cond2, label %Loop2, label %L2Exit

L2Exit:         ; preds = %L3Exit
        br i1 %cond1, label %Loop1, label %L1Exit

L1Exit:         ; preds = %L2Exit
        ret void
}

