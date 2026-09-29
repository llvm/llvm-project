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
; RUN: opt < %s -passes=instcombine
; END.

target datalayout = "e-p:32:32"
@silly = external constant i32          ; <ptr> [#uses=1]

declare void @bzero(ptr, i32)

declare void @bcopy(ptr, ptr, i32)

declare i32 @bcmp(ptr, ptr, i32)

declare i32 @fputs(ptr, ptr)

declare i32 @fputs_unlocked(ptr, ptr)

define i32 @function(i32 %a.1) {
entry:
        %a.0 = alloca i32               ; <ptr> [#uses=2]
        %result = alloca i32            ; <ptr> [#uses=2]
        store i32 %a.1, ptr %a.0
        %tmp.0 = load i32, ptr %a.0         ; <i32> [#uses=1]
        %tmp.1 = load i32, ptr @silly               ; <i32> [#uses=1]
        %tmp.2 = add i32 %tmp.0, %tmp.1         ; <i32> [#uses=1]
        store i32 %tmp.2, ptr %result
        br label %return

return:         ; preds = %entry
        %tmp.3 = load i32, ptr %result              ; <i32> [#uses=1]
        ret i32 %tmp.3
}

