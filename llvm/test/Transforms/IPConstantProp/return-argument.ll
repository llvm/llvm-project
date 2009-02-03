; RUN: llvm-as < %s | opt -ipconstprop | llvm-dis > %t
; RUN: cat %t | grep {store i32 %Z, i32\\* %Q}
; RUN: cat %t | grep {add i32 1, 3}

;; This function returns its second argument on all return statements
define internal i32* @incdec(i1 %C, i32* %V) {
        %X = load i32* %V
        br i1 %C, label %T, label %F

T:              ; preds = %0
        %X1 = add i32 %X, 1
        store i32 %X1, i32* %V
        ret i32* %V

F:              ; preds = %0
        %X2 = sub i32 %X, 1
        store i32 %X2, i32* %V
        ret i32* %V
}

;; This function returns its first argument as a part of a multiple return
;; value
define internal { i32, i32 } @foo(i32 %A, i32 %B) {
        %X = add i32 %A, %B
        %Y = insertvalue { i32, i32 } undef, i32 %A, 0
        %Z = insertvalue { i32, i32 } %Y, i32 %X, 1
        ret { i32, i32 } %Z
}

define void @caller(i1 %C) {
        %Q = alloca i32
        ;; Call incdec to see if %W is properly replaced by %Q
        %W = call i32* @incdec(i1 %C, i32* %Q )             ; <i32> [#uses=1]
        ;; Call @foo twice, to prevent the arguments from propagating into the
        ;; function (so we can check the returned argument is properly
        ;; propagated per-caller).
        %S1 = call { i32, i32 } @foo(i32 1, i32 2);
        %X1 = extractvalue { i32, i32 } %S1, 0
        %S2 = invoke { i32, i32 } @foo(i32 3, i32 4) to label %OK unwind label %RET;
OK:
        %X2 = extractvalue { i32, i32 } %S2, 0
        ;; Do some stuff with the returned values which we can grep for
        %Z  = add i32 %X1, %X2
        store i32 %Z, i32* %W
        br label %RET
RET:
        ret void
}

