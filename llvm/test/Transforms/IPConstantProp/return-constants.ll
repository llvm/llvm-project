; RUN: llvm-as < %s | opt -ipconstprop | llvm-dis > %t
;; Check that the 21 constants got propagated properly
; RUN: cat %t | grep {%M = add i32 21, 21}
;; Check that the second return values didn't get propagated
; RUN: cat %t | grep {%N = add i32 %B, %D}

define internal {i32, i32} @foo(i1 %Q) {
        br i1 %Q, label %T, label %F

T:              ; preds = %0
        ret i32 21, i32 22

F:              ; preds = %0
        ret i32 21, i32 23
}

define internal {i32, i32} @bar(i1 %Q) {
        %A = insertvalue { i32, i32 } undef, i32 21, 0
        br i1 %Q, label %T, label %F

T:              ; preds = %0
        %B = insertvalue { i32, i32 } %A, i32 22, 1
        ret { i32, i32 } %B

F:              ; preds = %0
        %C = insertvalue { i32, i32 } %A, i32 23, 1
        ret { i32, i32 } %C
}

define { i32, i32 } @caller(i1 %Q) {
        %X = call {i32, i32} @foo( i1 %Q )
        %A = getresult {i32, i32} %X, 0
        %B = getresult {i32, i32} %X, 1
        %Y = call {i32, i32} @bar( i1 %Q )
        %C = extractvalue {i32, i32} %Y, 0
        %D = extractvalue {i32, i32} %Y, 1
        %M = add i32 %A, %C
        %N = add i32 %B, %D
        ret { i32, i32 } %X
}

