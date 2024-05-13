; RUN: llc -mtriple=thumb-eabi %s -o /dev/null
; RUN: llc -mtriple=thumb-linux %s -o - | FileCheck %s
; RUN: llc -mtriple=thumb-darwin %s -o - | FileCheck %s

@str = internal constant [4 x i8] c"%d\0A\00"           ; <ptr> [#uses=1]

define void @f(i32 %a, ...) {
entry:
; Check that space is reserved above the pushed lr for variadic argument
; registers to be stored in.
; CHECK: sub sp, #[[IMM:[0-9]+]]
; CHECK: push
        %va = alloca ptr, align 4               ; <ptr> [#uses=4]
        call void @llvm.va_start( ptr %va )
        br label %bb

bb:             ; preds = %bb, %entry
        %a_addr.0 = phi i32 [ %a, %entry ], [ %tmp5, %bb ]              ; <i32> [#uses=2]
        %tmp = load volatile ptr, ptr %va           ; <ptr> [#uses=2]
        %tmp2 = getelementptr i8, ptr %tmp, i32 4           ; <ptr> [#uses=1]
        store volatile ptr %tmp2, ptr %va
        %tmp5 = add i32 %a_addr.0, -1           ; <i32> [#uses=1]
        %tmp.upgrd.2 = icmp eq i32 %a_addr.0, 1         ; <i1> [#uses=1]
        br i1 %tmp.upgrd.2, label %bb7, label %bb

bb7:            ; preds = %bb
        %tmp.upgrd.3 = load i32, ptr %tmp          ; <i32> [#uses=1]
        %tmp10 = call i32 (ptr, ...) @printf( ptr @str, i32 %tmp.upgrd.3 )                ; <i32> [#uses=0]
        call void @llvm.va_end( ptr %va )
        ret void

; The return sequence should pop the lr to r0-3, recover the stack space used to
; store variadic argument registers, then return via r3. Possibly there is a pop
; before this, but only if the function happened to use callee-saved registers.
; CHECK: pop {[[POP_REG:r[0-3]]]}
; CHECK: add sp, #[[IMM]]
; CHECK: bx [[POP_REG]]
}

declare void @llvm.va_start(ptr)

declare i32 @printf(ptr, ...)

declare void @llvm.va_end(ptr)
