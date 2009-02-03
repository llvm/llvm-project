; RUN: llvm-as < %s | \
; RUN:   llc -march=x86 -mtriple=i686-apple-darwin8 -relocation-model=static > %t
; RUN: grep {movl	_last} %t | count 1
; RUN: grep {cmpl.*_last} %t | count 1

@block = external global i8*            ; <i8**> [#uses=1]
@last = external global i32             ; <i32*> [#uses=3]

define i1 @loadAndRLEsource_no_exit_2E_1_label_2E_0(i32 %tmp.21.reload, i32 %tmp.8) {
newFuncRoot:
        br label %label.0
label.0.no_exit.1_crit_edge.exitStub:           ; preds = %label.0
        ret i1 true
codeRepl5.exitStub:             ; preds = %label.0
        ret i1 false
label.0:                ; preds = %newFuncRoot
        %tmp.35 = load i32* @last               ; <i32> [#uses=1]
        %inc.1 = add i32 %tmp.35, 1             ; <i32> [#uses=2]
        store i32 %inc.1, i32* @last
        %tmp.36 = load i8** @block              ; <i8*> [#uses=1]
        %tmp.38 = getelementptr i8* %tmp.36, i32 %inc.1         ; <i8*> [#uses=1]
        %tmp.40 = trunc i32 %tmp.21.reload to i8                ; <i8> [#uses=1]
        store i8 %tmp.40, i8* %tmp.38
        %tmp.910 = load i32* @last              ; <i32> [#uses=1]
        %tmp.1111 = icmp slt i32 %tmp.910, %tmp.8               ; <i1> [#uses=1]
        %tmp.1412 = icmp ne i32 %tmp.21.reload, 257             ; <i1> [#uses=1]
        %tmp.1613 = and i1 %tmp.1111, %tmp.1412         ; <i1> [#uses=1]
        br i1 %tmp.1613, label %label.0.no_exit.1_crit_edge.exitStub, label %codeRepl5.exitStub
}

