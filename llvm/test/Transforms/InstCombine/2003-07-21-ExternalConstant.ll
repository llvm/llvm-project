;
; Test: ExternalConstant
;
; Description:
;	This regression test helps check whether the instruction combining
;	optimization pass correctly handles global variables which are marked
;	as external and constant.
;
;	If a problem occurs, we should die on an assert().  Otherwise, we
;	should pass through the optimizer without failure.
;
; Extra code:
; RUN: llvm-as < %s | opt -instcombine
; END.

target datalayout = "e-p:32:32"
@silly = external constant i32          ; <i32*> [#uses=1]

declare void @bzero(i8*, i32)

declare void @bcopy(i8*, i8*, i32)

declare i32 @bcmp(i8*, i8*, i32)

declare i32 @fputs(i8*, i8*)

declare i32 @fputs_unlocked(i8*, i8*)

define i32 @function(i32 %a.1) {
entry:
        %a.0 = alloca i32               ; <i32*> [#uses=2]
        %result = alloca i32            ; <i32*> [#uses=2]
        store i32 %a.1, i32* %a.0
        %tmp.0 = load i32* %a.0         ; <i32> [#uses=1]
        %tmp.1 = load i32* @silly               ; <i32> [#uses=1]
        %tmp.2 = add i32 %tmp.0, %tmp.1         ; <i32> [#uses=1]
        store i32 %tmp.2, i32* %result
        br label %return

return:         ; preds = %entry
        %tmp.3 = load i32* %result              ; <i32> [#uses=1]
        ret i32 %tmp.3
}

