; RUN: llc -no-integrated-as < %s
; PR1308
; PR1557

; Bug: PR31336

define i32 @stuff(i32, ...) {
        %foo = alloca ptr
        %bar = alloca ptr
        %A = call i32 asm sideeffect "inline asm $0 $2 $3 $4", "=r,0,i,m,m"( i32 0, i32 1, ptr %foo, ptr %bar )
        ret i32 %A
}
